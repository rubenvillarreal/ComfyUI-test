/* modules/ui_components.js - Handles UI component creation and management */
import { DOMUtils, LogUtils } from "./utils.js";

export class UIComponents {
	constructor(node, renderer, imageManager, transformManager) {
		this.node = node;
		this.renderer = renderer;
		this.imageManager = imageManager;
		this.transformManager = transformManager;
	}

	// Create manual refresh button with debug info
	createRefreshButton() {
		const refreshButton = DOMUtils.createButton("Refresh Canvas", async () => {
			LogUtils.logGroup("Manual refresh triggered", {
				node: this.node,
				nodeImgs: this.node.imgs,
				widgets: this.node.widgets?.map(w => ({ name: w.name, type: w.type, value: w.value })),
				properties: this.node.properties,
				refImageSize: this.imageManager.getReferenceImageSize(),
				transformCache: this.transformManager.getTransformCache(),
				canvasDimensions: { 
					width: this.renderer.canvas.width, 
					height: this.renderer.canvas.height,
					rect: this.renderer.canvas.getBoundingClientRect()
				},
				coordinateSystem: this.renderer.calculateCoordinateSystem(),
				transformValues: this.transformManager.getAllProperties()
			});
			
			// Force refresh of transform data
			await this.transformManager.updateFromNode();
			await this.renderer.draw();
		});
		
		this.node.addDOMWidget("refresh_button", "div", refreshButton, {
			serialize: false,
			hideOnZoom: false,
		});
	}

	// Create debug info display
	createDebugInfo() {
		const debugInfo = DOMUtils.createInfoDiv("Images: 0/3 loaded");
		
		// Update debug info periodically
		setInterval(() => {
			const imageDebug = this.imageManager.getDebugInfo();
			const currentPose = this.renderer.state?.which || "A";
			const coords = this.node.properties ? 
				`${currentPose}: tx=${(this.transformManager.getProperty(`tx_${currentPose}`, 0)).toFixed(1)}, ty=${(this.transformManager.getProperty(`ty_${currentPose}`, 0)).toFixed(1)}` : "";
			const cacheStatus = this.transformManager.getTransformCache().matrices.A ? "Cached" : "No cache";
			const coordSys = this.renderer.calculateCoordinateSystem();
			const aspectRatioInfo = `Aspect: canvas=${(coordSys.actualCanvasWidth/coordSys.actualCanvasHeight).toFixed(2)}, ref=${(coordSys.refW/coordSys.refH).toFixed(2)}`;
			
			debugInfo.textContent = `Images: ${imageDebug.validCount}/3 | ${imageDebug.previewStatus} | node.imgs: ${imageDebug.nodeImgCount} | ${coords} | ${cacheStatus} | ${aspectRatioInfo}`;
		}, 1000);
		
		this.node.addDOMWidget("debug_info", "div", debugInfo, {
			serialize: false,
			hideOnZoom: false,
		});
	}

	// Create controls and coordinate system info
	createControlsInfo() {
		const coordInfo = document.createElement("div");
		coordInfo.style.marginTop = "5px";
		coordInfo.style.padding = "5px";
		coordInfo.style.backgroundColor = "#2a2a2a";
		coordInfo.style.color = "#aaa";
		coordInfo.style.fontSize = "9px";
		coordInfo.style.fontFamily = "monospace";
		coordInfo.style.borderRadius = "3px";
		
		const CANVAS_SIZE = 512; // Should match the constant from main widget
		coordInfo.innerHTML = `
			<strong>Controls:</strong><br>
			• Mouse: Drag to move poses<br>
			• Scroll: Scale active pose (↑=bigger, ↓=smaller)<br>
			• Shift+Scroll: Rotate active pose (↑=CW, ↓=CCW)<br>
			• A/B keys: Switch active pose<br>
			• R key: Reset active pose<br>
			• Arrow keys: Fine movement<br>
			<strong>Canvas:</strong><br>
			• Fixed ${CANVAS_SIZE}x${CANVAS_SIZE} resolution<br>
			• Maintains aspect ratio regardless of node size<br>
			• Coordinates match Python node exactly<br>
			<strong>Data Sources:</strong><br>
			• Auto mode: Uses Python matrices + offset correction<br>
			• Manual mode: Uses widget values + cached offset
		`;
		
		this.node.addDOMWidget("coord_info", "div", coordInfo, {
			serialize: false,
			hideOnZoom: false,
		});
	}

	// Create a status indicator component
	createStatusIndicator() {
		const statusDiv = document.createElement("div");
		statusDiv.style.marginTop = "5px";
		statusDiv.style.padding = "3px 8px";
		statusDiv.style.backgroundColor = "#333";
		statusDiv.style.color = "#fff";
		statusDiv.style.fontSize = "11px";
		statusDiv.style.fontFamily = "monospace";
		statusDiv.style.borderRadius = "3px";
		statusDiv.style.textAlign = "center";
		
		const updateStatus = () => {
			const imageDebug = this.imageManager.getDebugInfo();
			const hasImages = imageDebug.hasValidImages;
			const cacheAvailable = this.transformManager.getTransformCache().matrices.A !== null;
			
			if (!hasImages) {
				statusDiv.style.backgroundColor = "#666";
				statusDiv.textContent = "No Images - Run Workflow";
			} else if (cacheAvailable) {
				statusDiv.style.backgroundColor = "#006600";
				statusDiv.textContent = "Auto Mode - Python Transforms";
			} else {
				statusDiv.style.backgroundColor = "#0066cc";
				statusDiv.textContent = "Manual Mode - Widget Controls";
			}
		};
		
		// Update status periodically
		setInterval(updateStatus, 1000);
		updateStatus(); // Initial update
		
		this.node.addDOMWidget("status_indicator", "div", statusDiv, {
			serialize: false,
			hideOnZoom: false,
		});
	}

	// Create all UI components
	createAllComponents() {
		this.createRefreshButton();
		this.createDebugInfo();
		this.createStatusIndicator();
		this.createControlsInfo();
	}
}