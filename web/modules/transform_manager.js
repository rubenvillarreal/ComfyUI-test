/* modules/transform_manager.js - Handles transformation data and property management */
import { api } from "../../../scripts/api.js";

export class TransformManager {
	constructor(node) {
		this.node = node;
		this.lastProperties = {};
		this.transformCache = {
			lastUpdate: 0,
			matrices: { A: null, B: null },
			offsetCorrections: { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } }
		};
		this.propertyMonitorInterval = null;
	}

	// Normalize angle to 0-360 degrees range
	normalizeAngle(angle) {
		return ((angle % 360) + 360) % 360;
	}

	// Set a property value on both the node and its widget
	setProperty(key, value) {
		// Normalize angle values
		if (key.includes('angle_deg_')) {
			value = this.normalizeAngle(value);
		}
		
		if (this.node.properties[key] !== value) {
			this.node.setProperty(key, value);
			const widget = this.node.widgets.find(w => w.name === key);
			if (widget) { 
				widget.value = value;
				// Trigger widget callback to ensure UI updates
				if (widget.callback) {
					widget.callback(value);
				}
			}
		}
	}

	// Get a property value safely with default
	getProperty(key, defaultValue = 0) {
		return this.node.properties && this.node.properties[key] !== undefined ? 
			this.node.properties[key] : defaultValue;
	}

	// Get all current transformation properties
	getAllProperties() {
		return {
			tx_A: this.getProperty('tx_A'),
			ty_A: this.getProperty('ty_A'),
			scale_A: this.getProperty('scale_A', 1),
			angle_deg_A: this.getProperty('angle_deg_A'),
			tx_B: this.getProperty('tx_B'),
			ty_B: this.getProperty('ty_B'),
			scale_B: this.getProperty('scale_B', 1),
			angle_deg_B: this.getProperty('angle_deg_B')
		};
	}

	// Check if transformation properties have changed
	checkPropertiesChanged() {
		const currentProps = this.getAllProperties();
		
		const changed = Object.keys(currentProps).some(key => 
			this.lastProperties[key] !== currentProps[key]
		);

		if (changed) {
			this.lastProperties = { ...currentProps };
			return true;
		}
		return false;
	}

	// Build affine transformation matrix exactly like the Python node
	buildAffineMatrix(scale, angleDeg, tx, ty, cx, cy) {
		const angleRad = angleDeg * Math.PI / 180;
		const cosA = Math.cos(angleRad);
		const sinA = Math.sin(angleRad);
		
		// This matches the Python _build_affine function exactly
		const R11 = cosA * scale;
		const R12 = -sinA * scale;
		const R21 = sinA * scale;
		const R22 = cosA * scale;
		
		// Translation with rotation center compensation
		const finalTx = tx + cx - (R11 * cx + R12 * cy);
		const finalTy = ty + cy - (R21 * cx + R22 * cy);
		
		return {
			a: R11,     // scale * cos(angle)
			b: R21,     // scale * sin(angle)  
			c: R12,     // scale * -sin(angle)
			d: R22,     // scale * cos(angle)
			e: finalTx, // x translation
			f: finalTy  // y translation
		};
	}

	// Get current transformation for a pose (A or B)
	getCurrentTransform(keyPrefix, refW, refH) {
		let tx, ty, scale, rotD;
		
		// Check if we have cached matrix data from Python (automatic mode)
		if (this.transformCache.matrices[keyPrefix]) {
			// Use data from Python node (includes offset correction)
			const matrix = this.transformCache.matrices[keyPrefix];
			
			// Convert 2x3 matrix back to transform parameters
			const cx = refW / 2.0;
			const cy = refH / 2.0;
			
			// Extract scale and rotation from matrix
			scale = Math.sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1]);
			rotD = Math.atan2(matrix[1], matrix[0]) * 180 / Math.PI;
			
			// Extract translation (already includes offset correction)
			tx = matrix[4];
			ty = matrix[5];
			
			console.log(`Using cached matrix for ${keyPrefix}:`, { matrix, scale, rotD, tx, ty });
		} else {
			// Fallback to widget values (manual mode)
			tx = this.getProperty(`tx_${keyPrefix}`, 0);
			ty = this.getProperty(`ty_${keyPrefix}`, 0);
			scale = this.getProperty(`scale_${keyPrefix}`, 1);
			rotD = this.getProperty(`angle_deg_${keyPrefix}`, 0);
			
			// Apply offset correction if available
			const offset = this.transformCache.offsetCorrections[keyPrefix];
			if (offset) {
				tx += offset.x;
				ty += offset.y;
			}
			
			console.log(`Using widget values for ${keyPrefix}:`, { tx, ty, scale, rotD, offset });
		}

		// Build the affine matrix using reference image center
		const cx = refW / 2.0;
		const cy = refH / 2.0;
		const matrix = this.buildAffineMatrix(scale, rotD, tx, ty, cx, cy);

		return { tx, ty, scale, rotD, matrix };
	}

	// Get transformation data from Python node via API
	async updateFromNode() {
		try {
			// Check if node has updated transformation data
			const nodeId = this.node.id;
			const response = await api.fetchApi(`/AInseven/pose_align_data/${nodeId}`);
			
			if (response.ok) {
				const data = await response.json();
				if (data.timestamp > this.transformCache.lastUpdate) {
					this.transformCache = {
						lastUpdate: data.timestamp,
						matrices: data.matrices || { A: null, B: null },
						offsetCorrections: data.offsetCorrections || { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } }
					};
					console.log("Updated transform cache from node:", this.transformCache);
					return true; // Data was updated
				}
			}
		} catch (error) {
			console.log("No transform data available from node (this is normal before first execution)");
		}
		return false; // No update
	}

	// Get current transform cache
	getTransformCache() {
		return this.transformCache;
	}

	// Set up monitoring for property changes and automatic updates
	setupPropertyMonitoring(renderer) {
		// Monitor all transformation widgets for value changes
		const monitoredParams = ['tx_A', 'ty_A', 'scale_A', 'angle_deg_A', 
		                         'tx_B', 'ty_B', 'scale_B', 'angle_deg_B'];
		
		// Add value change listeners to widgets
		const setupWidgetMonitoring = () => {
			monitoredParams.forEach(paramName => {
				const widget = this.node.widgets?.find(w => w.name === paramName);
				if (widget) {
					const originalCallback = widget.callback;
					widget.callback = function(value) {
						// Call original callback if exists
						if (originalCallback) {
							originalCallback.call(this, value);
						}
						// Update canvas
						console.log(`Widget ${paramName} changed to ${value}`);
						setTimeout(() => renderer.draw(), 10); // Small delay to ensure property is updated
					};
				}
			});
		};

		// Setup monitoring after widgets are created
		setTimeout(setupWidgetMonitoring, 100);

		// Periodic property change monitoring and transform data updates
		this.propertyMonitorInterval = setInterval(async () => {
			// Check for updated transform data from Python
			const dataUpdated = await this.updateFromNode();
			
			// Check for property changes
			const propsChanged = this.checkPropertiesChanged();
			
			if (dataUpdated || propsChanged) {
				console.log("Properties or transform data changed, redrawing canvas");
				renderer.draw();
			}
		}, 500); // Check every 500ms
	}

	// Clean up monitoring when node is removed
	cleanup() {
		if (this.propertyMonitorInterval) {
			clearInterval(this.propertyMonitorInterval);
			this.propertyMonitorInterval = null;
		}
	}
}