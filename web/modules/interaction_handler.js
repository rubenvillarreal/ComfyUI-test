/* modules/interaction_handler.js - Handles mouse and keyboard interactions */

export class InteractionHandler {
	constructor(canvas, transformManager, renderer) {
		this.canvas = canvas;
		this.transformManager = transformManager;
		this.renderer = renderer;
		this.state = null; // Will be set by main widget
	}

	setState(state) {
		this.state = state;
	}

	// Set up all event listeners
	setupEventListeners() {
		this.setupMouseEvents();
		this.setupKeyboardEvents();
	}

	// Set up mouse event listeners
	setupMouseEvents() {
		// Prevent context menu
		this.canvas.addEventListener("contextmenu", e => e.preventDefault());
		
		// Mouse down - start dragging
		this.canvas.addEventListener("mousedown", (e) => {
			this.state.dragging = true;
			this.state.which = e.button === 2 ? "B" : "A"; // Right click = B, Left click = A
			const coords = this.renderer.getCanvasCoordinates(e);
			this.state.lastX = coords.x; 
			this.state.lastY = coords.y;
			this.canvas.style.cursor = "grabbing";
			this.renderer.draw();
		});
		
		// Mouse up - stop dragging
		this.canvas.addEventListener("mouseup", () => {
			this.state.dragging = false;
			this.canvas.style.cursor = "crosshair";
		});
		
		// Mouse leave - stop dragging and hovering
		this.canvas.addEventListener("mouseleave", () => {
			this.state.dragging = false;
			this.state.hovering = false;
			this.canvas.style.cursor = "crosshair";
		});
		
		// Mouse enter - start hovering
		this.canvas.addEventListener("mouseenter", () => {
			this.state.hovering = true;
		});
		
		// Mouse move - handle dragging
		this.canvas.addEventListener("mousemove", (e) => {
			if (!this.state.dragging) return;
			
			const coords = this.renderer.getCanvasCoordinates(e);
			const dx = coords.x - this.state.lastX;
			const dy = coords.y - this.state.lastY;
			this.state.lastX = coords.x; 
			this.state.lastY = coords.y;
			
			// Convert canvas pixel movement to reference image coordinate movement
			const coordSys = this.renderer.calculateCoordinateSystem();
			const { canvasScale } = coordSys;
			
			// Scale the movement to match the reference coordinate system
			const scaledDx = dx / canvasScale;
			const scaledDy = dy / canvasScale;
			
			const pose = this.state.which;
			const currentTx = this.transformManager.getProperty(`tx_${pose}`, 0);
			const currentTy = this.transformManager.getProperty(`ty_${pose}`, 0);
			
			this.transformManager.setProperty(`tx_${pose}`, currentTx + scaledDx);
			this.transformManager.setProperty(`ty_${pose}`, currentTy + scaledDy);
			this.renderer.draw();
		});
		
		// Mouse wheel - handle scaling and rotation
		this.canvas.addEventListener("wheel", (e) => {
			e.preventDefault();
			const pose = this.state.which;
			
			if (e.shiftKey) { 
				// Rotation with shift key
				const rotationStep = e.deltaY > 0 ? 5 : -5; // Reversed for intuitive direction
				const currentAngle = this.transformManager.getProperty(`angle_deg_${pose}`, 0);
				const newAngle = currentAngle + rotationStep;
				this.transformManager.setProperty(`angle_deg_${pose}`, newAngle);
			} else { 
				// Scaling without shift key
				const currentScale = this.transformManager.getProperty(`scale_${pose}`, 1);
				const scaleStep = e.deltaY > 0 ? -0.05 : 0.05; // Reversed for intuitive direction
				const newScale = Math.max(0.1, currentScale + scaleStep);
				this.transformManager.setProperty(`scale_${pose}`, newScale);
			}
			this.renderer.draw();
		});
	}

	// Set up keyboard event listeners
	setupKeyboardEvents() {
		this.canvas.addEventListener("keydown", (e) => {
			if (!this.state.hovering) return;
			
			// Calculate step size based on reference image coordinate system
			const coordSys = this.renderer.calculateCoordinateSystem();
			const { canvasScale } = coordSys;
			const step = (e.shiftKey ? 10 : 1) / canvasScale; // Convert to reference coordinates
			const pose = this.state.which;
			
			switch(e.key) {
				case 'ArrowLeft':
					e.preventDefault();
					this.adjustProperty(`tx_${pose}`, -step);
					break;
				case 'ArrowRight':
					e.preventDefault();
					this.adjustProperty(`tx_${pose}`, step);
					break;
				case 'ArrowUp':
					e.preventDefault();
					this.adjustProperty(`ty_${pose}`, -step);
					break;
				case 'ArrowDown':
					e.preventDefault();
					this.adjustProperty(`ty_${pose}`, step);
					break;
				case 'a':
				case 'A':
					e.preventDefault();
					this.state.which = "A";
					this.renderer.draw();
					break;
				case 'b':
				case 'B':
					e.preventDefault();
					this.state.which = "B";
					this.renderer.draw();
					break;
				case 'r':
				case 'R':
					e.preventDefault();
					this.resetCurrentPose();
					break;
			}
		});
	}

	// Helper method to adjust a property by a delta value
	adjustProperty(propertyName, delta) {
		const currentValue = this.transformManager.getProperty(propertyName, 0);
		this.transformManager.setProperty(propertyName, currentValue + delta);
		this.renderer.draw();
	}

	// Reset the current active pose to default values
	resetCurrentPose() {
		const pose = this.state.which;
		this.transformManager.setProperty(`tx_${pose}`, 0);
		this.transformManager.setProperty(`ty_${pose}`, 0);
		this.transformManager.setProperty(`scale_${pose}`, 1.0);
		this.transformManager.setProperty(`angle_deg_${pose}`, 0);
		this.renderer.draw();
	}
}