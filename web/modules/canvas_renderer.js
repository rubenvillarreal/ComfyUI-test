/* modules/canvas_renderer.js - Handles all canvas drawing operations */

export class CanvasRenderer {
	constructor(canvas, ctx, imageManager, transformManager) {
		this.canvas = canvas;
		this.ctx = ctx;
		this.imageManager = imageManager;
		this.transformManager = transformManager;
		this.state = null; // Will be set by main widget
	}

	setState(state) {
		this.state = state;
	}

	// Calculate proper coordinate system for drawing
	calculateCoordinateSystem() {
		const actualCanvasWidth = this.canvas.width;
		const actualCanvasHeight = this.canvas.height;
		
		const refSize = this.imageManager.getReferenceImageSize();
		const refW = refSize.width;
		const refH = refSize.height;
		
		const canvasAspect = actualCanvasWidth / actualCanvasHeight;
		const refAspect = refW / refH;
		
		let canvasScale, offsetX, offsetY;
		
		if (refAspect > canvasAspect) {
			// Reference is wider than canvas - fit to width
			canvasScale = actualCanvasWidth / refW * 0.9;
			offsetX = actualCanvasWidth * 0.05;
			offsetY = (actualCanvasHeight - refH * canvasScale) / 2;
		} else {
			// Reference is taller than canvas - fit to height
			canvasScale = actualCanvasHeight / refH * 0.9;
			offsetY = actualCanvasHeight * 0.05;
			offsetX = (actualCanvasWidth - refW * canvasScale) / 2;
		}
		
		return {
			canvasScale,
			offsetX,
			offsetY,
			refW,
			refH,
			actualCanvasWidth,
			actualCanvasHeight
		};
	}

	// Draw subtle grid background
	drawGrid(coordSys) {
		const { canvasScale, offsetX, offsetY, actualCanvasWidth, actualCanvasHeight } = coordSys;
		
		this.ctx.strokeStyle = "#333";
		this.ctx.lineWidth = 1;
		this.ctx.setLineDash([2, 4]);
		
		const gridStep = 32 * canvasScale;
		
		// Vertical lines
		for (let i = offsetX; i < actualCanvasWidth; i += gridStep) {
			this.ctx.beginPath();
			this.ctx.moveTo(i, 0);
			this.ctx.lineTo(i, actualCanvasHeight);
			this.ctx.stroke();
		}
		
		// Horizontal lines
		for (let i = offsetY; i < actualCanvasHeight; i += gridStep) {
			this.ctx.beginPath();
			this.ctx.moveTo(0, i);
			this.ctx.lineTo(actualCanvasWidth, i);
			this.ctx.stroke();
		}
		
		this.ctx.setLineDash([]);
	}

	// Generate placeholder pose visualization
	generatePoseVisualization(type, width = 384, height = 384) {
		const canvas = document.createElement('canvas');
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext('2d');
		
		// Dark background
		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, width, height);
		
		// Draw placeholder pose stick figure
		ctx.strokeStyle = type === 'ref' ? '#666' : 
		                 type === 'A' ? '#ff4a4a' : '#4a9eff';
		ctx.lineWidth = 3;
		ctx.lineCap = 'round';
		
		const centerX = width / 2;
		const centerY = height / 2;
		const scale = 50;
		
		// Simple stick figure
		// Head
		ctx.beginPath();
		ctx.arc(centerX, centerY - scale, 15, 0, Math.PI * 2);
		ctx.stroke();
		
		// Body
		ctx.beginPath();
		ctx.moveTo(centerX, centerY - scale + 15);
		ctx.lineTo(centerX, centerY + scale);
		ctx.stroke();
		
		// Arms
		ctx.beginPath();
		ctx.moveTo(centerX - scale * 0.8, centerY - scale * 0.3);
		ctx.lineTo(centerX + scale * 0.8, centerY - scale * 0.3);
		ctx.stroke();
		
		// Legs
		ctx.beginPath();
		ctx.moveTo(centerX, centerY + scale);
		ctx.lineTo(centerX - scale * 0.5, centerY + scale * 1.5);
		ctx.moveTo(centerX, centerY + scale);
		ctx.lineTo(centerX + scale * 0.5, centerY + scale * 1.5);
		ctx.stroke();
		
		// Add label
		ctx.fillStyle = ctx.strokeStyle;
		ctx.font = '12px monospace';
		ctx.textAlign = 'center';
		const label = type === 'ref' ? 'REF (placeholder)' : 
		             type === 'A' ? 'POSE A (placeholder)' : 'POSE B (placeholder)';
		ctx.fillText(label, centerX, height - 20);
		
		return canvas;
	}

	// Draw a single pose layer
	drawPoseLayer(img, keyPrefix, color, isActive, coordSys, hasValidImages) {
		const { canvasScale, offsetX, offsetY, refW, refH } = coordSys;
		
		// If no image, use placeholder if no valid images exist
		if (!img) {
			if (!hasValidImages) {
				const type = keyPrefix === "REF" ? "ref" :
				            keyPrefix === "A" ? "A" : "B";
				img = this.generatePoseVisualization(type, refW, refH);
			} else {
				console.log(`No image for ${keyPrefix}`);
				return;
			}
		}

		// Reset transform for this layer
		this.ctx.setTransform(1, 0, 0, 1, 0, 0);

		if (keyPrefix === "REF") {
			// Reference image: just scale and center it
			this.ctx.globalAlpha = 0.5;
			this.ctx.drawImage(img, offsetX, offsetY, refW * canvasScale, refH * canvasScale);
			console.log(`Drew reference pose at: ${offsetX}, ${offsetY}, ${refW * canvasScale}x${refH * canvasScale}`);
		} else {
			// For poses A and B, get current transform values
			const transform = this.transformManager.getCurrentTransform(keyPrefix, refW, refH);
			const { tx, ty, scale, rotD, matrix } = transform;
			
			console.log(`Pose ${keyPrefix} transform:`, { tx, ty, scale, rotD, matrix });

			// Apply proper coordinate system transformation
			this.ctx.setTransform(
				matrix.a * canvasScale,  // scale x component
				matrix.b * canvasScale,  // skew y component
				matrix.c * canvasScale,  // skew x component
				matrix.d * canvasScale,  // scale y component
				matrix.e * canvasScale + offsetX,  // translate x
				matrix.f * canvasScale + offsetY   // translate y
			);

			// Draw the pose image with proper aspect ratio
			this.ctx.globalAlpha = 0.9;
			this.ctx.drawImage(img, 0, 0, refW, refH);
			console.log(`Drew pose ${keyPrefix} with transform`);
			
			// Reset transform to draw outline
			this.ctx.setTransform(1, 0, 0, 1, 0, 0);
			
			// Calculate transformed corners for outline
			const corners = [
				[0, 0], [refW, 0], [refW, refH], [0, refH]
			].map(([x, y]) => [
				(matrix.a * x + matrix.c * y + matrix.e) * canvasScale + offsetX,
				(matrix.b * x + matrix.d * y + matrix.f) * canvasScale + offsetY
			]);

			// Draw colored outline
			this.ctx.globalAlpha = isActive ? 1.0 : 0.6;
			this.ctx.strokeStyle = color;
			this.ctx.lineWidth = isActive ? 3 : 2;
			this.ctx.setLineDash(isActive ? [] : [5, 5]);
			this.ctx.beginPath();
			this.ctx.moveTo(corners[0][0], corners[0][1]);
			for (let i = 1; i < corners.length; i++) {
				this.ctx.lineTo(corners[i][0], corners[i][1]);
			}
			this.ctx.closePath();
			this.ctx.stroke();
			this.ctx.setLineDash([]);
		}
	}

	// Draw UI overlay elements
	drawUIOverlay(coordSys, hasValidImages) {
		const { actualCanvasWidth, actualCanvasHeight } = coordSys;
		
		// Reset transform for UI elements
		this.ctx.setTransform(1, 0, 0, 1, 0, 0);
		this.ctx.globalAlpha = 1.0;
		
		// Show status if no valid images
		if (!hasValidImages) {
			this.ctx.fillStyle = "#888";
			this.ctx.font = "14px monospace";
			this.ctx.textAlign = "center";
			this.ctx.fillText("Run workflow to generate images", actualCanvasWidth/2, actualCanvasHeight/2 - 20);
			
			this.ctx.fillStyle = "#666";
			this.ctx.font = "12px monospace";
			this.ctx.fillText("Images will appear after node execution", actualCanvasWidth/2, actualCanvasHeight/2 + 5);
			
			// Show debug info
			this.ctx.font = "10px monospace";
			const nodeImgCount = this.imageManager.node.imgs?.length || 0;
			this.ctx.fillText(`node.imgs found: ${nodeImgCount}`, actualCanvasWidth/2, actualCanvasHeight/2 + 25);
		}
		
		// Active pose indicator in top-left
		this.ctx.fillStyle = this.state.which === "A" ? "#ff4a4a" : "#4a9eff";
		this.ctx.fillRect(10, 10, 20, 20);
		this.ctx.strokeStyle = "#fff";
		this.ctx.lineWidth = 2;
		this.ctx.strokeRect(10, 10, 20, 20);
		
		this.ctx.fillStyle = "#fff";
		this.ctx.font = "12px monospace";
		this.ctx.textAlign = "left";
		this.ctx.fillText(`Active: Pose ${this.state.which}`, 40, 25);
		
		// Instructions in bottom-left
		this.ctx.fillStyle = "#aaa";
		this.ctx.font = "10px monospace";
		this.ctx.fillText("Left-click: Move Pose A", 10, actualCanvasHeight - 70);
		this.ctx.fillText("Right-click: Move Pose B", 10, actualCanvasHeight - 55);
		this.ctx.fillText("Wheel: Scale | Shift+Wheel: Rotate", 10, actualCanvasHeight - 40);
		this.ctx.fillText("Arrow keys: Fine movement | R: Reset pose", 10, actualCanvasHeight - 25);
		
		const refSize = this.imageManager.getReferenceImageSize();
		const canvasScale = coordSys.canvasScale;
		this.ctx.fillText(`Ref: ${refSize.width}x${refSize.height} | Scale: ${canvasScale.toFixed(3)} | Canvas: ${actualCanvasWidth}x${actualCanvasHeight}`, 10, actualCanvasHeight - 10);
	}

	// Main drawing function
	async draw() {
		try {
			console.log("Starting draw function...");
			
			// Try to get updated transform data from Python node
			const dataUpdated = await this.transformManager.updateFromNode();
			if (dataUpdated) {
				console.log("Transform data updated from Python node");
			}
			
			// Get images from the node
			const images = await this.imageManager.getImagesFromNode();
			
			// Check if we have any actual images
			const hasValidImages = !!(images.ref || images.A || images.B);
			
			// Calculate proper coordinate system
			const coordSys = this.calculateCoordinateSystem();
			const { actualCanvasWidth, actualCanvasHeight } = coordSys;
			
			// Debug logging
			console.log("Drawing poses:", { 
				ref: !!images.ref, 
				A: !!images.A, 
				B: !!images.B,
				hasValidImages,
				refImageSize: this.imageManager.getReferenceImageSize(),
				canvasActualSize: { width: actualCanvasWidth, height: actualCanvasHeight },
				canvasScale: coordSys.canvasScale,
				offset: { x: coordSys.offsetX, y: coordSys.offsetY },
				nodeImgsLength: this.imageManager.node.imgs?.length || 0,
				properties: this.transformManager.getAllProperties(),
				transformCache: this.transformManager.getTransformCache()
			});
			
			// Clear and set up canvas
			this.ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
			this.ctx.clearRect(0, 0, actualCanvasWidth, actualCanvasHeight);
			
			// Dark background
			this.ctx.fillStyle = "#1a1a1a";
			this.ctx.fillRect(0, 0, actualCanvasWidth, actualCanvasHeight);
			
			// Draw grid
			this.drawGrid(coordSys);

			// Draw layers in order: reference, B, then A on top
			this.drawPoseLayer(images.ref, "REF", "#666", false, coordSys, hasValidImages);
			this.drawPoseLayer(images.B, "B", "#4a9eff", this.state.which === "B", coordSys, hasValidImages);
			this.drawPoseLayer(images.A, "A", "#ff4a4a", this.state.which === "A", coordSys, hasValidImages);
			
			// Draw UI overlay
			this.drawUIOverlay(coordSys, hasValidImages);
			
		} catch (error) {
			console.error("Error in draw function:", error);
			
			// Draw error message
			this.ctx.setTransform(1, 0, 0, 1, 0, 0);
			this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
			this.ctx.fillStyle = "#1a1a1a";
			this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
			this.ctx.fillStyle = "#ff4444";
			this.ctx.font = "14px monospace";
			this.ctx.textAlign = "center";
			this.ctx.fillText("Error loading images", this.canvas.width/2, this.canvas.height/2);
			this.ctx.font = "10px monospace";
			this.ctx.fillText("Check console for details", this.canvas.width/2, this.canvas.height/2 + 20);
		}
	}

	// Get canvas coordinates from mouse event
	getCanvasCoordinates(e) {
		const rect = this.canvas.getBoundingClientRect();
		return {
			x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
			y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
		};
	}
}