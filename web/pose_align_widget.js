/* pose_align_widget_preview_access.js – Canvas that properly accesses stored preview images
    This script accesses images stored by PreviewImage mechanism
*/
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "AInseven.PoseAlignCanvasWidget.PreviewAccess",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Exit if this is not the node we want to modify
		if (nodeData.name !== "PoseAlignTwoToOne") return;

		// --- Hijack the onNodeCreated method to add our custom widget ---
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			// Call the original onNodeCreated method
			onNodeCreated?.apply(this, arguments);

			const node = this; // Use a clearer variable for 'this'

			// --- Create the Canvas Element ---
			const canvas = document.createElement("canvas");
			canvas.width = 384;
			canvas.height = 384;
			canvas.style.border = "2px solid #555";
			canvas.style.backgroundColor = "#1a1a1a";
			canvas.style.display = "block";
			canvas.style.cursor = "crosshair";
			canvas.style.borderRadius = "4px";
			
			const ctx = canvas.getContext("2d");

			// --- State management for interaction ---
			const st = { 
				dragging: false, 
				which: "A", 
				lastX: 0, 
				lastY: 0,
				hovering: false,
				loadedImages: {
					ref: null,
					A: null,
					B: null
				},
				previewImage: null
			};

			// --- Helper: Update node properties ---
			function setVal(key, v) {
				if (node.properties[key] !== v) {
					node.setProperty(key, v);
					const w = node.widgets.find(w => w.name === key);
					if (w) { w.value = v; }
				}
			}

			// --- Helper: Extract individual poses from combined preview image ---
			async function extractPosesFromPreview(previewImage) {
				if (!previewImage) return { ref: null, A: null, B: null };
				
				// The preview image has all three poses stacked horizontally
				const width = previewImage.width / 3;
				const height = previewImage.height;
				
				// Create canvases for each pose
				const poses = {};
				const names = ['ref', 'A', 'B'];
				
				for (let i = 0; i < 3; i++) {
					const tempCanvas = document.createElement('canvas');
					tempCanvas.width = width;
					tempCanvas.height = height;
					const tempCtx = tempCanvas.getContext('2d');
					
					// Draw the portion of the preview image
					tempCtx.drawImage(previewImage, 
						i * width, 0, width, height,  // source
						0, 0, width, height            // destination
					);
					
					// Convert to image bitmap for use
					poses[names[i]] = await createImageBitmap(tempCanvas);
				}
				
				return poses;
			}

			// --- Proper Helper: Get images from node's execution results ---
			async function getImagesFromNode() {
				try {
					// Check if we have a preview image from the node's UI output
					if (node.imgs && node.imgs.length > 0 && node.imgs[0].src) {
						console.log("Found preview image in node.imgs:", node.imgs[0]);
						
						// Load the preview image
						const img = new Image();
						img.src = node.imgs[0].src;
						await new Promise((resolve, reject) => {
							img.onload = resolve;
							img.onerror = reject;
						});
						
						st.previewImage = img;
						
						// Extract individual poses from the combined preview
						const poses = await extractPosesFromPreview(img);
						console.log("Extracted poses from preview image");
						return poses;
					}
					
					// Alternative: Check for images in the node's widgets
					const imageWidget = node.widgets?.find(w => w.type === "image");
					if (imageWidget && imageWidget.value) {
						console.log("Found image in widget:", imageWidget);
						
						const filename = imageWidget.value.filename || imageWidget.value;
						const subfolder = imageWidget.value.subfolder || "";
						const type = imageWidget.value.type || "output";
						
						const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
						console.log("Loading preview from widget URL:", url);
						
						const response = await fetch(url);
						if (response.ok) {
							const blob = await response.blob();
							const img = await createImageBitmap(blob);
							st.previewImage = img;
							
							// Extract individual poses
							const poses = await extractPosesFromPreview(img);
							console.log("Extracted poses from widget image");
							return poses;
						}
					}
					
					console.log("No images found in node");
					return { ref: null, A: null, B: null };
					
				} catch (error) {
					console.error("Error getting images from node:", error);
					return { ref: null, A: null, B: null };
				}
			}

			// --- Generate placeholder pose visualization ---
			function generatePoseVisualization(type, width = 384, height = 384) {
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

			// --- Main Drawing Function ---
			const draw = async () => {
				try {
					console.log("Starting draw function...");
					
					// Get images from the node
					const images = await getImagesFromNode();
					
					// Update loaded images state
					st.loadedImages = images;
					
					// Check if we have any actual images
					const hasValidImages = !!(images.ref || images.A || images.B);
					
					// Debug logging
					console.log("Drawing poses:", { 
						ref: !!images.ref, 
						A: !!images.A, 
						B: !!images.B,
						hasValidImages,
						nodeImgsLength: node.imgs?.length || 0
					});
					
					ctx.clearRect(0, 0, canvas.width, canvas.height);
					
					// Dark background with subtle grid
					ctx.fillStyle = "#1a1a1a";
					ctx.fillRect(0, 0, canvas.width, canvas.height);
					
					// Draw subtle grid for reference
					ctx.strokeStyle = "#333";
					ctx.lineWidth = 1;
					ctx.setLineDash([2, 4]);
					for (let i = 0; i < canvas.width; i += 32) {
						ctx.beginPath();
						ctx.moveTo(i, 0);
						ctx.lineTo(i, canvas.height);
						ctx.stroke();
					}
					for (let i = 0; i < canvas.height; i += 32) {
						ctx.beginPath();
						ctx.moveTo(0, i);
						ctx.lineTo(canvas.width, i);
						ctx.stroke();
					}
					ctx.setLineDash([]);

					// Helper function to draw pose with transformations
					function drawPoseLayer(img, keyPrefix, color, isActive = false) {
						// If no image, use placeholder if no valid images exist
						if (!img) {
							if (!hasValidImages) {
								const type = keyPrefix === "REF" ? "ref" :
											keyPrefix === "A" ? "A" : "B";
								img = generatePoseVisualization(type);
							} else {
								console.log(`No image for ${keyPrefix}`);
								return;
							}
						}
						
						const tx = node.properties[`tx_${keyPrefix}`] ?? 0;
						const ty = node.properties[`ty_${keyPrefix}`] ?? 0;
						const sc = node.properties[`scale_${keyPrefix}`] ?? 1;
						const rotD = node.properties[`angle_deg_${keyPrefix}`] ?? 0;

						ctx.save();
						ctx.translate(canvas.width / 2 + tx, canvas.height / 2 + ty);
						ctx.rotate(rotD * Math.PI / 180);
						ctx.scale(sc, sc);

						// Scale image to fit canvas better
						const maxSize = Math.min(canvas.width, canvas.height) * 0.8;
						const scale = Math.min(maxSize / img.width, maxSize / img.height);
						const drawWidth = img.width * scale;
						const drawHeight = img.height * scale;

						// For reference pose, draw with transparency
						if (keyPrefix === "REF") {
							ctx.globalAlpha = 0.5;
							ctx.drawImage(img, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
							console.log(`Drew reference pose: ${drawWidth}x${drawHeight}`);
						} else {
							// For poses A and B
							ctx.globalAlpha = 0.9;
							ctx.drawImage(img, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
							console.log(`Drew pose ${keyPrefix}: ${drawWidth}x${drawHeight}`);
							
							// Add colored outline for better visibility
							ctx.globalAlpha = isActive ? 1.0 : 0.6;
							ctx.strokeStyle = color;
							ctx.lineWidth = isActive ? 3 : 2;
							ctx.setLineDash(isActive ? [] : [5, 5]);
							ctx.strokeRect(-drawWidth / 2 - 2, -drawHeight / 2 - 2, drawWidth + 4, drawHeight + 4);
							ctx.setLineDash([]);
						}
						
						ctx.restore();
					}
					
					// Draw layers in order: reference, B, then A on top
					drawPoseLayer(images.ref, "REF", "#666");
					drawPoseLayer(images.B, "B", "#4a9eff", st.which === "B"); // Blue for pose B
					drawPoseLayer(images.A, "A", "#ff4a4a", st.which === "A"); // Red for pose A
					
					// Draw UI indicators
					ctx.globalAlpha = 1.0;
					
					// Show status
					if (!hasValidImages) {
						ctx.fillStyle = "#888";
						ctx.font = "14px monospace";
						ctx.textAlign = "center";
						ctx.fillText("Run workflow to generate images", canvas.width/2, canvas.height/2 - 20);
						
						ctx.fillStyle = "#666";
						ctx.font = "12px monospace";
						ctx.fillText("Images will appear after node execution", canvas.width/2, canvas.height/2 + 5);
						
						// Show debug info
						ctx.font = "10px monospace";
						ctx.fillText(`node.imgs found: ${node.imgs?.length || 0}`, canvas.width/2, canvas.height/2 + 25);
					}
					
					// Active pose indicator in top-left
					ctx.fillStyle = st.which === "A" ? "#ff4a4a" : "#4a9eff";
					ctx.fillRect(10, 10, 20, 20);
					ctx.strokeStyle = "#fff";
					ctx.lineWidth = 2;
					ctx.strokeRect(10, 10, 20, 20);
					
					ctx.fillStyle = "#fff";
					ctx.font = "12px monospace";
					ctx.textAlign = "left";
					ctx.fillText(`Active: Pose ${st.which}`, 40, 25);
					
					// Instructions in bottom-left
					ctx.fillStyle = "#aaa";
					ctx.font = "10px monospace";
					ctx.fillText("Left-click: Move Pose A", 10, canvas.height - 30);
					ctx.fillText("Right-click: Move Pose B", 10, canvas.height - 15);
					ctx.fillText("Wheel: Scale | Shift+Wheel: Rotate", 10, canvas.height - 5);
					
				} catch (error) {
					console.error("Error in draw function:", error);
					
					// Draw error message
					ctx.clearRect(0, 0, canvas.width, canvas.height);
					ctx.fillStyle = "#1a1a1a";
					ctx.fillRect(0, 0, canvas.width, canvas.height);
					ctx.fillStyle = "#ff4444";
					ctx.font = "14px monospace";
					ctx.textAlign = "center";
					ctx.fillText("Error loading images", canvas.width/2, canvas.height/2);
					ctx.font = "10px monospace";
					ctx.fillText("Check console for details", canvas.width/2, canvas.height/2 + 20);
				}
			};

			// --- Mouse & Wheel Event Listeners ---
			canvas.addEventListener("contextmenu", e => e.preventDefault());
			
			canvas.addEventListener("mousedown", e => {
				st.dragging = true;
				st.which = e.button === 2 ? "B" : "A";
				st.lastX = e.offsetX; 
				st.lastY = e.offsetY;
				canvas.style.cursor = "grabbing";
				draw();
			});
			
			canvas.addEventListener("mouseup", () => {
				st.dragging = false;
				canvas.style.cursor = "crosshair";
			});
			
			canvas.addEventListener("mouseleave", () => {
				st.dragging = false;
				st.hovering = false;
				canvas.style.cursor = "crosshair";
			});
			
			canvas.addEventListener("mouseenter", () => {
				st.hovering = true;
			});
			
			canvas.addEventListener("mousemove", e => {
				if (!st.dragging) return;
				
				const dx = e.offsetX - st.lastX;
				const dy = e.offsetY - st.lastY;
				st.lastX = e.offsetX; 
				st.lastY = e.offsetY;
				
				const p = st.which;
				setVal(`tx_${p}`, (node.properties[`tx_${p}`] || 0) + dx);
				setVal(`ty_${p}`, (node.properties[`ty_${p}`] || 0) + dy);
				draw();
			});
			
			canvas.addEventListener("wheel", e => {
				e.preventDefault();
				const p = st.which;
				
				if (e.shiftKey) { 
					const rotationStep = e.deltaY < 0 ? -5 : 5;
					setVal(`angle_deg_${p}`, (node.properties[`angle_deg_${p}`] || 0) + rotationStep);
				} else { 
					const cur = node.properties[`scale_${p}`] || 1;
					const scaleStep = e.deltaY < 0 ? 0.05 : -0.05;
					setVal(`scale_${p}`, Math.max(0.1, cur + scaleStep));
				}
				draw();
			});

			// --- Add keyboard shortcuts ---
			canvas.addEventListener("keydown", e => {
				if (!st.hovering) return;
				
				const step = e.shiftKey ? 10 : 1;
				const p = st.which;
				
				switch(e.key) {
					case 'ArrowLeft':
						e.preventDefault();
						setVal(`tx_${p}`, (node.properties[`tx_${p}`] || 0) - step);
						draw();
						break;
					case 'ArrowRight':
						e.preventDefault();
						setVal(`tx_${p}`, (node.properties[`tx_${p}`] || 0) + step);
						draw();
						break;
					case 'ArrowUp':
						e.preventDefault();
						setVal(`ty_${p}`, (node.properties[`ty_${p}`] || 0) - step);
						draw();
						break;
					case 'ArrowDown':
						e.preventDefault();
						setVal(`ty_${p}`, (node.properties[`ty_${p}`] || 0) + step);
						draw();
						break;
					case 'a':
					case 'A':
						e.preventDefault();
						st.which = "A";
						draw();
						break;
					case 'b':
					case 'B':
						e.preventDefault();
						st.which = "B";
						draw();
						break;
					case 'r':
					case 'R':
						e.preventDefault();
						// Reset current pose
						setVal(`tx_${p}`, 0);
						setVal(`ty_${p}`, 0);
						setVal(`scale_${p}`, 1.0);
						setVal(`angle_deg_${p}`, 0);
						draw();
						break;
				}
			});
			
			// Make canvas focusable for keyboard events
			canvas.tabIndex = 0;

			// --- Add the canvas as a DOM widget to the node's body ---
			node.addDOMWidget("pose_canvas", "div", canvas, {
				serialize: false,
				hideOnZoom: false,
			});

			// --- Override onExecuted to handle UI updates ---
			const originalOnExecuted = node.onExecuted;
			node.onExecuted = function(message) {
				originalOnExecuted?.apply(this, arguments);
				console.log("Node executed, message:", message);
				
				// The PreviewImage system returns data in message.images
				if (message && message.images && message.images.length > 0) {
					console.log("Found images in execution message:", message.images);
					// The image is already loaded into node.imgs by ComfyUI
					setTimeout(draw, 100);
				} else {
					console.log("No images in execution message");
					// Still try to draw in case images are loaded differently
					setTimeout(draw, 500);
				}
			};

			// --- Alternative: Hook into the node's image widget updates ---
			const checkForImageWidget = () => {
				const imageWidget = node.widgets?.find(w => w.type === "image" || w.name === "images");
				if (imageWidget) {
					console.log("Found image widget:", imageWidget);
					
					const originalCallback = imageWidget.callback;
					imageWidget.callback = function() {
						originalCallback?.apply(this, arguments);
						console.log("Image widget updated");
						setTimeout(draw, 100);
					};
					
					// Also check for value changes
					const originalSetValue = imageWidget.setValue;
					if (originalSetValue) {
						imageWidget.setValue = function(value) {
							originalSetValue.call(this, value);
							console.log("Image widget value set:", value);
							setTimeout(draw, 100);
						};
					}
				}
			};
			
			// Check for image widget after a delay
			setTimeout(checkForImageWidget, 100);

			const onConnectionsChange = node.onConnectionsChange;
			node.onConnectionsChange = function() {
				onConnectionsChange?.apply(this, arguments);
				console.log("Connections changed");
				setTimeout(draw, 100);
			};

			const onPropertyChanged = node.onPropertyChanged;
			node.onPropertyChanged = function(name, value) {
				onPropertyChanged?.apply(this, arguments);
				if (name.includes('tx_') || name.includes('ty_') || 
				    name.includes('scale_') || name.includes('angle_deg_')) {
					draw();
				}
			};

			// Initial draw
			setTimeout(draw, 500);
			
			// Add manual refresh button with debug info
			const refreshButton = document.createElement("button");
			refreshButton.textContent = "Refresh Canvas";
			refreshButton.style.marginTop = "5px";
			refreshButton.style.padding = "5px 10px";
			refreshButton.style.backgroundColor = "#444";
			refreshButton.style.color = "#fff";
			refreshButton.style.border = "1px solid #666";
			refreshButton.style.borderRadius = "3px";
			refreshButton.style.cursor = "pointer";
			refreshButton.onclick = async () => {
				console.log("Manual refresh triggered");
				console.log("Current node:", node);
				console.log("Node.imgs:", node.imgs);
				console.log("Node widgets:", node.widgets?.map(w => ({ name: w.name, type: w.type, value: w.value })));
				
				// Try to find any image data
				if (node.imgs && node.imgs.length > 0) {
					console.log("Preview images found:", node.imgs);
				}
				
				// Check all widgets for image data
				for (const widget of (node.widgets || [])) {
					if (widget.element && widget.element.tagName === 'IMG') {
						console.log("Found IMG element in widget:", widget.name, widget.element.src);
					}
				}
				
				await draw();
			};
			
			node.addDOMWidget("refresh_button", "div", refreshButton, {
				serialize: false,
				hideOnZoom: false,
			});

			// Add debug info display
			const debugInfo = document.createElement("div");
			debugInfo.style.marginTop = "5px";
			debugInfo.style.padding = "5px";
			debugInfo.style.backgroundColor = "#2a2a2a";
			debugInfo.style.color = "#ccc";
			debugInfo.style.fontSize = "10px";
			debugInfo.style.fontFamily = "monospace";
			debugInfo.style.borderRadius = "3px";
			debugInfo.textContent = "Images: 0/3 loaded";
			
			// Update debug info periodically
			setInterval(() => {
				const imgCount = node.imgs?.length || 0;
				const validCount = st.loadedImages ? 
					(st.loadedImages.ref ? 1 : 0) + (st.loadedImages.A ? 1 : 0) + (st.loadedImages.B ? 1 : 0) : 0;
				const previewStatus = st.previewImage ? "Preview loaded" : "No preview";
				debugInfo.textContent = `Images: ${validCount}/3 displayed | ${previewStatus} | node.imgs: ${imgCount}`;
			}, 1000);
			
			node.addDOMWidget("debug_info", "div", debugInfo, {
				serialize: false,
				hideOnZoom: false,
			});
		};
	}
});
