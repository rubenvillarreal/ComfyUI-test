/* modules/utils.js - Shared utility functions */

export class MathUtils {
	// Normalize angle to 0-360 degrees range
	static normalizeAngle(angle) {
		return ((angle % 360) + 360) % 360;
	}

	// Clamp a value between min and max
	static clamp(value, min, max) {
		return Math.min(Math.max(value, min), max);
	}

	// Linear interpolation between two values
	static lerp(start, end, factor) {
		return start + (end - start) * factor;
	}

	// Convert degrees to radians
	static degToRad(degrees) {
		return degrees * Math.PI / 180;
	}

	// Convert radians to degrees
	static radToDeg(radians) {
		return radians * 180 / Math.PI;
	}
}

export class DOMUtils {
	// Create a styled button element
	static createButton(text, onClick, styles = {}) {
		const button = document.createElement("button");
		button.textContent = text;
		button.onclick = onClick;
		
		// Default button styles
		const defaultStyles = {
			marginTop: "5px",
			padding: "5px 10px",
			backgroundColor: "#444",
			color: "#fff",
			border: "1px solid #666",
			borderRadius: "3px",
			cursor: "pointer"
		};
		
		// Apply styles
		Object.assign(button.style, defaultStyles, styles);
		
		return button;
	}

	// Create a styled info div
	static createInfoDiv(content, styles = {}) {
		const div = document.createElement("div");
		
		// Default info div styles
		const defaultStyles = {
			marginTop: "5px",
			padding: "5px",
			backgroundColor: "#2a2a2a",
			color: "#ccc",
			fontSize: "10px",
			fontFamily: "monospace",
			borderRadius: "3px"
		};
		
		// Apply styles
		Object.assign(div.style, defaultStyles, styles);
		
		if (typeof content === 'string') {
			div.innerHTML = content;
		} else {
			div.textContent = content;
		}
		
		return div;
	}

	// Get element coordinates relative to its container
	static getRelativeCoordinates(element, event) {
		const rect = element.getBoundingClientRect();
		return {
			x: (event.clientX - rect.left) * (element.width / rect.width),
			y: (event.clientY - rect.top) * (element.height / rect.height)
		};
	}
}

export class AsyncUtils {
	// Create a promise that resolves after a specified delay
	static delay(ms) {
		return new Promise(resolve => setTimeout(resolve, ms));
	}

	// Debounce a function call
	static debounce(func, delay) {
		let timeoutId;
		return function (...args) {
			clearTimeout(timeoutId);
			timeoutId = setTimeout(() => func.apply(this, args), delay);
		};
	}

	// Throttle a function call
	static throttle(func, delay) {
		let lastCall = 0;
		return function (...args) {
			const now = Date.now();
			if (now - lastCall >= delay) {
				lastCall = now;
				return func.apply(this, args);
			}
		};
	}
}

export class ValidationUtils {
	// Validate that a value is a number within range
	static validateNumber(value, min = -Infinity, max = Infinity, defaultValue = 0) {
		const num = parseFloat(value);
		if (isNaN(num)) return defaultValue;
		return MathUtils.clamp(num, min, max);
	}

	// Validate transformation parameters
	static validateTransformParams(params) {
		return {
			tx: ValidationUtils.validateNumber(params.tx, -2048, 2048, 0),
			ty: ValidationUtils.validateNumber(params.ty, -2048, 2048, 0),
			scale: ValidationUtils.validateNumber(params.scale, 0.1, 3.0, 1.0),
			angle: MathUtils.normalizeAngle(ValidationUtils.validateNumber(params.angle, -720, 720, 0))
		};
	}
}

export class LogUtils {
	static logGroup(title, data) {
		console.group(`[PoseAlign] ${title}`);
		console.log(data);
		console.groupEnd();
	}

	static logError(context, error) {
		console.error(`[PoseAlign] Error in ${context}:`, error);
	}

	static logDebug(message, data = null) {
		console.log(`[PoseAlign] ${message}`, data || '');
	}
}