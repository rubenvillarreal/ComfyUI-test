/* pose_align_widget_fixed.js – Fixed aspect ratio independence from node dimensions */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "AInseven.PoseAlignCanvasWidget.Fixed",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Exit if this is not the node we want to modify
		if (nodeData.name !== "PoseAlignTwoToOne") return;

		// --- Hijack the onNodeCreated method to add our custom widget ---
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			// Call the original onNodeCreated method
			onNodeCreated?.apply(this, arguments);

			const node = this; // Use a clearer variable for 'this'

			// --- Create the Canvas Element with FIXED aspect ratio ---
			const canvas = document.createElement("canvas");
			
			// FIXED: Use a fixed canvas size that maintains 1:1 aspect ratio
			// This will be independent of the node's UI dimensions
			const CANVAS_SIZE = 512; // Fixed square canvas
			canvas.width = CANVAS_SIZE;
			canvas.height = CANVAS_SIZE;
			
			canvas.style.border = "2px solid #555";
			canvas.style.backgroundColor = "#1a1a1a";
			canvas.style.display = "block";
			canvas.style.cursor = "crosshair";
			canvas.style.borderRadius = "4px";
			
			// FIXED: Ensure canvas maintains its aspect ratio regardless of container
			canvas.style.width = `${CANVAS_SIZE}px`;
			canvas.style.height = `${CANVAS_SIZE}px`;
			canvas.style.minWidth = `${CANVAS_SIZE}px`;
			canvas.style.minHeight = `${CANVAS_SIZE}px`;
			canvas.style.maxWidth = `${CANVAS_SIZE}px`;
			canvas.style.maxHeight = `${CANVAS_SIZE}px`;
			canvas.style.objectFit = "contain";
			canvas.style.flexShrink = "0";
			
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
				previewImage: null,
				// Store the reference image dimensions (the target coordinate system)
				refImageSize: { width: 512, height: 512 },
				// Track last property values to detect changes
				lastProperties: {},
				// Cache for transformation data from Python node
				transformCache: {
					lastUpdate: 0,
					matrices: { A: null, B: null },
					offsetCorrections: { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } }
				}
			};

			// --- Helper: Get transformation data from Python node via temp file ---
			async function getTransformDataFromNode() {
				try {
					// Check if node has updated transformation data
					const nodeId = node.id;
					const response = await api.fetchApi(`/AInseven/pose_align_data/${nodeId}`);
					
					if (response.ok) {
						const data = await response.json();
						if (data.timestamp > st.transformCache.lastUpdate) {
							st.transformCache = {
								lastUpdate: data.timestamp,
								matrices: data.matrices || { A: null, B: null },
								offsetCorrections: data.offsetCorrections || { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } }
							};
							console.log("Updated transform cache from node:", st.transformCache);
							return true; // Data was updated
						}
					}
				} catch (error) {
					console.log("No transform data available from node (this is normal before first execution)");
				}
				return false; // No update
			}

			// --- Helper: Update node properties AND widgets ---
			function setVal(key, v) {
				// Normalize angle values
				if (key.includes('angle_deg_')) {
					v = ((v % 360) + 360) % 360; // Normalize to 0-360
				}
				
				if (node.properties[key] !== v) {
					node.setProperty(key, v);
					const w = node.widgets.find(w => w.name === key);
					if (w) { 
						w.value = v;
						// Trigger widget callback to ensure UI updates
						if (w.callback) {
							w.callback(v);
						}
					}
				}
			}

			// --- Helper: Get current property values safely ---
			function getProperty(key, defaultValue = 0) {
				return node.properties && node.properties[key] !== undefined ? node.properties[key] : defaultValue;
			}

			// --- Helper: Check if transformation properties have changed ---
			function checkPropertiesChanged() {
				const currentProps = {
					tx_A: getProperty('tx_A'),
					ty_A: getProperty('ty_A'),
					scale_A: getProperty('scale_A', 1),
					angle_deg_A: getProperty('angle_deg_A'),
					tx_B: getProperty('tx_B'),
					ty_B: getProperty('ty_B'),
					scale_B: getProperty('scale_B', 1),
					angle_deg_B: getProperty('angle_deg_B')
				};

				const changed = Object.keys(currentProps).some(key => 
					st.lastProperties[key] !== currentProps[key]
				);

				if (changed) {
					st.lastProperties = { ...currentProps };
					return true;
				}
				return false;
			}

			// --- Helper: Build affine transformation matrix exactly like the node ---
			function buildAffineMatrix(scale, angleDeg, tx, ty, cx, cy) {
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
					a: R11,  // scale * cos(angle)
					b: R21,  // scale * sin(angle)  
					c: R12,  // scale * -sin(angle)
					d: R22,  // scale * cos(angle)
					e: finalTx, // x translation
					f: finalTy  // y translation
				};
			}

			// --- Helper: Apply offset correction to transformation ---
			function applyOffsetCorrection(matrix, offsetCorrection) {
				return {
					a: matrix.a,
					b: matrix.b,
					c: matrix.c,
					d: matrix.d,
					e: matrix.e + offsetCorrection.x,
					f: matrix.f + offsetCorrection.y
				};
			}

			// --- Helper: Extract individual poses from combined preview image ---
			async function extractPosesFromPreview(previewImage) {
				if (!previewImage) return { ref: null, A: null, B: null };
				
				// The preview image has all three poses stacked horizontally
				const width = previewImage.width / 3;
				const height = previewImage.height;
				
				// Update reference image size for coordinate system
				st.refImageSize = { width, height };
				
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

			// --- Get images from node's execution results ---
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
						console.log("Extracted poses from preview image, ref size:", st.refImageSize);
						return poses;
					}
					
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

			// --- FIXED: Coordinate system calculation ---
			function calculateCoordinateSystem() {
				// Get actual canvas dimensions (might be different from initial size due to CSS)
				const canvasRect = canvas.getBoundingClientRect();
				const actualCanvasWidth = canvas.width;  // Use actual canvas resolution
				const actualCanvasHeight = canvas.height;
				
				// Get reference image dimensions
				const refW = st.refImageSize.width;
				const refH = st.refImageSize.height;
				
				// Calculate aspect ratios
				const canvasAspect = actualCanvasWidth / actualCanvasHeight;
				const refAspect = refW / refH;
				
				// Calculate scale to fit reference image in canvas while maintaining aspect ratio
				let canvasScale;
				let offsetX, offsetY;
				
				if (refAspect > canvasAspect) {
					// Reference is wider than canvas - fit to width
					canvasScale = actualCanvasWidth / refW * 0.9; // 90% to leave some margin
					offsetX = actualCanvasWidth * 0.05; // 5% margin on each side
					offsetY = (actualCanvasHeight - refH * canvasScale) / 2;
				} else {
					// Reference is taller than canvas - fit to height
					canvasScale = actualCanvasHeight / refH * 0.9; // 90% to leave some margin
					offsetY = actualCanvasHeight * 0.05; // 5% margin on top/bottom
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

			// --- Main Drawing Function ---
			const draw = async () => {
				try {
					console.log("Starting draw function...");
					
					// Try to get updated transform data from Python node
					const dataUpdated = await getTransformDataFromNode();
					if (dataUpdated) {
						console.log("Transform data updated from Python node");
					}
					
					// Get images from the node
					const images = await getImagesFromNode();
					
					// Update loaded images state
					st.loadedImages = images;
					
					// Check if we have any actual images
					const hasValidImages = !!(images.ref || images.A || images.B);
					
					// FIXED: Calculate proper coordinate system
					const coordSys = calculateCoordinateSystem();
					const { canvasScale, offsetX, offsetY, refW, refH, actualCanvasWidth, actualCanvasHeight } = coordSys;
					
					// Debug logging
					console.log("Drawing poses:", { 
						ref: !!images.ref, 
						A: !!images.A, 
						B: !!images.B,
						hasValidImages,
						refImageSize: st.refImageSize,
						canvasActualSize: { width: actualCanvasWidth, height: actualCanvasHeight },
						canvasScale,
						offset: { x: offsetX, y: offsetY },
						nodeImgsLength: node.imgs?.length || 0,
						properties: {
							tx_A: getProperty('tx_A'),
							ty_A: getProperty('ty_A'),
							scale_A: getProperty('scale_A', 1),
							angle_deg_A: getProperty('angle_deg_A'),
							tx_B: getProperty('tx_B'),
							ty_B: getProperty('ty_B'),
							scale_B: getProperty('scale_B', 1),
							angle_deg_B: getProperty('angle_deg_B')
						},
						transformCache: st.transformCache
					});
					
					// Clear and set up canvas
					ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
					ctx.clearRect(0, 0, actualCanvasWidth, actualCanvasHeight);
					
					// Dark background with subtle grid
					ctx.fillStyle = "#1a1a1a";
					ctx.fillRect(0, 0, actualCanvasWidth, actualCanvasHeight);
					
					// Draw subtle grid for reference (aligned to reference coordinate system)
					ctx.strokeStyle = "#333";
					ctx.lineWidth = 1;
					ctx.setLineDash([2, 4]);
					const gridStep = 32 * canvasScale; // Grid aligned to reference coordinates
					for (let i = offsetX; i < actualCanvasWidth; i += gridStep) {
						ctx.beginPath();
						ctx.moveTo(i, 0);
						ctx.lineTo(i, actualCanvasHeight);
						ctx.stroke();
					}
					for (let i = offsetY; i < actualCanvasHeight; i += gridStep) {
						ctx.beginPath();
						ctx.moveTo(0, i);
						ctx.lineTo(actualCanvasWidth, i);
						ctx.stroke();
					}
					ctx.setLineDash([]);

					// Helper function to draw pose with exact coordinate system matching
					function drawPoseLayer(img, keyPrefix, color, isActive = false) {
						// If no image, use placeholder if no valid images exist
						if (!img) {
							if (!hasValidImages) {
								const type = keyPrefix === "REF" ? "ref" :
											keyPrefix === "A" ? "A" : "B";
								img = generatePoseVisualization(type, refW, refH);
							} else {
								console.log(`No image for ${keyPrefix}`);
								return;
							}
						}

						// Reset transform for this layer
						ctx.setTransform(1, 0, 0, 1, 0, 0);

						if (keyPrefix === "REF") {
							// Reference image: just scale and center it
							ctx.globalAlpha = 0.5;
							ctx.drawImage(img, offsetX, offsetY, refW * canvasScale, refH * canvasScale);
							console.log(`Drew reference pose at: ${offsetX}, ${offsetY}, ${refW * canvasScale}x${refH * canvasScale}`);
						} else {
							// For poses A and B, get current transform values
							let tx, ty, sc, rotD;
							
							// Check if we have cached matrix data from Python (automatic mode)
							if (st.transformCache.matrices[keyPrefix]) {
								// Use data from Python node (includes offset correction)
								const matrix = st.transformCache.matrices[keyPrefix];
								
								// Convert 2x3 matrix back to transform parameters
								const cx = refW / 2.0;
								const cy = refH / 2.0;
								
								// Extract scale and rotation from matrix
								sc = Math.sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1]);
								rotD = Math.atan2(matrix[1], matrix[0]) * 180 / Math.PI;
								
								// Extract translation (already includes offset correction)
								tx = matrix[4];
								ty = matrix[5];
								
								console.log(`Using cached matrix for ${keyPrefix}:`, { matrix, sc, rotD, tx, ty });
							} else {
								// Fallback to widget values (manual mode)
								tx = getProperty(`tx_${keyPrefix}`, 0);
								ty = getProperty(`ty_${keyPrefix}`, 0);
								sc = getProperty(`scale_${keyPrefix}`, 1);
								rotD = getProperty(`angle_deg_${keyPrefix}`, 0);
								
								// Apply offset correction if available
								const offset = st.transformCache.offsetCorrections[keyPrefix];
								if (offset) {
									tx += offset.x;
									ty += offset.y;
								}
								
								console.log(`Using widget values for ${keyPrefix}:`, { tx, ty, sc, rotD, offset });
							}

							// Build the affine matrix using reference image center
							const cx = refW / 2.0;
							const cy = refH / 2.0;
							const affineMatrix = buildAffineMatrix(sc, rotD, tx, ty, cx, cy);

							console.log(`Pose ${keyPrefix} transform:`, { tx, ty, sc, rotD, cx, cy, affineMatrix });

							// FIXED: Apply proper coordinate system transformation
							// The key is to apply the canvas coordinate system transformation correctly
							ctx.setTransform(
								affineMatrix.a * canvasScale,  // scale x component
								affineMatrix.b * canvasScale,  // skew y component
								affineMatrix.c * canvasScale,  // skew x component
								affineMatrix.d * canvasScale,  // scale y component
								affineMatrix.e * canvasScale + offsetX,  // translate x
								affineMatrix.f * canvasScale + offsetY   // translate y
							);

							// Draw the pose image with proper aspect ratio
							ctx.globalAlpha = 0.9;
							ctx.drawImage(img, 0, 0, refW, refH);
							console.log(`Drew pose ${keyPrefix} with transform`);
							
							// Reset transform to draw outline
							ctx.setTransform(1, 0, 0, 1, 0, 0);
							
							// Calculate transformed corners for outline
							const corners = [
								[0, 0], [refW, 0], [refW, refH], [0, refH]
							].map(([x, y]) => [
								(affineMatrix.a * x + affineMatrix.c * y + affineMatrix.e) * canvasScale + offsetX,
								(affineMatrix.b * x + affineMatrix.d * y + affineMatrix.f) * canvasScale + offsetY
							]);

							// Draw colored outline
							ctx.globalAlpha = isActive ? 1.0 : 0.6;
							ctx.strokeStyle = color;
							ctx.lineWidth = isActive ? 3 : 2;
							ctx.setLineDash(isActive ? [] : [5, 5]);
							ctx.beginPath();
							ctx.moveTo(corners[0][0], corners[0][1]);
							for (let i = 1; i < corners.length; i++) {
								ctx.lineTo(corners[i][0], corners[i][1]);
							}
							ctx.closePath();
							ctx.stroke();
							ctx.setLineDash([]);
						}
					}
					
					// Draw layers in order: reference, B, then A on top
					drawPoseLayer(images.ref, "REF", "#666");
					drawPoseLayer(images.B, "B", "#4a9eff", st.which === "B"); // Blue for pose B
					drawPoseLayer(images.A, "A", "#ff4a4a", st.which === "A"); // Red for pose A
					
					// Reset transform for UI elements
					ctx.setTransform(1, 0, 0, 1, 0, 0);
					ctx.globalAlpha = 1.0;
					
					// Show status
					if (!hasValidImages) {
						ctx.fillStyle = "#888";
						ctx.font = "14px monospace";
						ctx.textAlign = "center";
						ctx.fillText("Run workflow to generate images", actualCanvasWidth/2, actualCanvasHeight/2 - 20);
						
						ctx.fillStyle = "#666";
						ctx.font = "12px monospace";
						ctx.fillText("Images will appear after node execution", actualCanvasWidth/2, actualCanvasHeight/2 + 5);
						
						// Show debug info
						ctx.font = "10px monospace";
						ctx.fillText(`node.imgs found: ${node.imgs?.length || 0}`, actualCanvasWidth/2, actualCanvasHeight/2 + 25);
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
					ctx.fillText("Left-click: Move Pose A", 10, actualCanvasHeight - 70);
					ctx.fillText("Right-click: Move Pose B", 10, actualCanvasHeight - 55);
					ctx.fillText("Wheel: Scale | Shift+Wheel: Rotate", 10, actualCanvasHeight - 40);
					ctx.fillText("Arrow keys: Fine movement | R: Reset pose", 10, actualCanvasHeight - 25);
					ctx.fillText(`Ref: ${refW}x${refH} | Scale: ${canvasScale.toFixed(3)} | Canvas: ${actualCanvasWidth}x${actualCanvasHeight}`, 10, actualCanvasHeight - 10);
					
				} catch (error) {
					console.error("Error in draw function:", error);
					
					// Draw error message
					ctx.setTransform(1, 0, 0, 1, 0, 0);
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

			// --- FIXED: Mouse event coordinate conversion ---
			function getCanvasCoordinates(e) {
				const rect = canvas.getBoundingClientRect();
				return {
					x: (e.clientX - rect.left) * (canvas.width / rect.width),
					y: (e.clientY - rect.top) * (canvas.height / rect.height)
				};
			}

			// --- Mouse & Wheel Event Listeners ---
			canvas.addEventListener("contextmenu", e => e.preventDefault());
			
			canvas.addEventListener("mousedown", e => {
				st.dragging = true;
				st.which = e.button === 2 ? "B" : "A";
				const coords = getCanvasCoordinates(e);
				st.lastX = coords.x; 
				st.lastY = coords.y;
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
				
				const coords = getCanvasCoordinates(e);
				const dx = coords.x - st.lastX;
				const dy = coords.y - st.lastY;
				st.lastX = coords.x; 
				st.lastY = coords.y;
				
				// FIXED: Convert canvas pixel movement to reference image coordinate movement
				const coordSys = calculateCoordinateSystem();
				const { canvasScale } = coordSys;
				
				// Scale the movement to match the reference coordinate system
				const scaledDx = dx / canvasScale;
				const scaledDy = dy / canvasScale;
				
				const p = st.which;
				setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) + scaledDx);
				setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) + scaledDy);
				draw();
			});
			
			canvas.addEventListener("wheel", e => {
				e.preventDefault();
				const p = st.which;
				
				if (e.shiftKey) { 
					// Fixed: bidirectional rotation with proper angle wrapping
					const rotationStep = e.deltaY > 0 ? 5 : -5; // Reversed for intuitive direction
					const currentAngle = getProperty(`angle_deg_${p}`, 0);
					const newAngle = currentAngle + rotationStep;
					setVal(`angle_deg_${p}`, newAngle); // setVal will handle normalization
				} else { 
					// Fixed: bidirectional scaling
					const cur = getProperty(`scale_${p}`, 1);
					const scaleStep = e.deltaY > 0 ? -0.05 : 0.05; // Reversed for intuitive direction
					setVal(`scale_${p}`, Math.max(0.1, cur + scaleStep));
				}
				draw();
			});

			// --- Add keyboard shortcuts ---
			canvas.addEventListener("keydown", e => {
				if (!st.hovering) return;
				
				// FIXED: Scale step size based on reference image coordinate system
				const coordSys = calculateCoordinateSystem();
				const { canvasScale } = coordSys;
				const step = (e.shiftKey ? 10 : 1) / canvasScale; // Convert to reference coordinates
				const p = st.which;
				
				switch(e.key) {
					case 'ArrowLeft':
						e.preventDefault();
						setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) - step);
						draw();
						break;
					case 'ArrowRight':
						e.preventDefault();
						setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) + step);
						draw();
						break;
					case 'ArrowUp':
						e.preventDefault();
						setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) - step);
						draw();
						break;
					case 'ArrowDown':
						e.preventDefault();
						setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) + step);
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
			// FIXED: Wrap canvas in a container to control sizing
			const canvasContainer = document.createElement("div");
			canvasContainer.style.display = "flex";
			canvasContainer.style.justifyContent = "center";
			canvasContainer.style.alignItems = "center";
			canvasContainer.style.width = "100%";
			canvasContainer.style.minHeight = `${CANVAS_SIZE}px`;
			canvasContainer.style.overflow = "visible";
			canvasContainer.appendChild(canvas);

			node.addDOMWidget("pose_canvas", "div", canvasContainer, {
				serialize: false,
				hideOnZoom: false,
			});

			// --- FIXED: Proper widget value change monitoring ---
			// Monitor all transformation widgets for value changes
			const monitoredParams = ['tx_A', 'ty_A', 'scale_A', 'angle_deg_A', 
			                         'tx_B', 'ty_B', 'scale_B', 'angle_deg_B'];
			
			// Add value change listeners to widgets
			const setupWidgetMonitoring = () => {
				monitoredParams.forEach(paramName => {
					const widget = node.widgets?.find(w => w.name === paramName);
					if (widget) {
						const originalCallback = widget.callback;
						widget.callback = function(value) {
							// Call original callback if exists
							if (originalCallback) {
								originalCallback.call(this, value);
							}
							// Update canvas
							console.log(`Widget ${paramName} changed to ${value}`);
							setTimeout(draw, 10); // Small delay to ensure property is updated
						};
					}
				});
			};

			// Setup monitoring after widgets are created
			setTimeout(setupWidgetMonitoring, 100);

			// --- Periodic property change monitoring and transform data updates ---
			let propertyMonitorInterval = setInterval(async () => {
				// Check for updated transform data from Python
				const dataUpdated = await getTransformDataFromNode();
				
				// Check for property changes
				const propsChanged = checkPropertiesChanged();
				
				if (dataUpdated || propsChanged) {
					console.log("Properties or transform data changed, redrawing canvas");
					draw();
				}
			}, 500); // Check every 500ms

			// Clean up interval when node is removed
			const originalOnRemoved = node.onRemoved;
			node.onRemoved = function() {
				if (propertyMonitorInterval) {
					clearInterval(propertyMonitorInterval);
					propertyMonitorInterval = null;
				}
				originalOnRemoved?.apply(this, arguments);
			};

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

				// Force a property check after execution (automatic mode may have updated them)
				setTimeout(async () => {
					const dataUpdated = await getTransformDataFromNode();
					const propsChanged = checkPropertiesChanged();
					if (dataUpdated || propsChanged) {
						console.log("Properties or transform data updated after execution, redrawing");
						draw();
					}
				}, 1000);
			};

			// --- FIXED: Property change monitoring ---
			const originalOnPropertyChanged = node.onPropertyChanged;
			node.onPropertyChanged = function(name, value) {
				originalOnPropertyChanged?.apply(this, arguments);
				
				// Redraw if any transformation parameter changes
				if (monitoredParams.includes(name)) {
					console.log(`Property ${name} changed to ${value}`);
					draw();
				}
			};

			// --- Connection change monitoring ---
			const onConnectionsChange = node.onConnectionsChange;
			node.onConnectionsChange = function() {
				onConnectionsChange?.apply(this, arguments);
				console.log("Connections changed");
				setTimeout(draw, 100);
			};

			// --- FIXED: Window resize handling ---
			const handleResize = () => {
				console.log("Window resized, redrawing canvas");
				setTimeout(draw, 100);
			};
			window.addEventListener('resize', handleResize);
			
			// Clean up resize listener when node is removed
			const originalOnRemovedResize = node.onRemoved;
			node.onRemoved = function() {
				window.removeEventListener('resize', handleResize);
				originalOnRemovedResize?.apply(this, arguments);
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
				console.log("Node properties:", node.properties);
				console.log("Reference image size:", st.refImageSize);
				console.log("Transform cache:", st.transformCache);
				console.log("Canvas dimensions:", { 
					width: canvas.width, 
					height: canvas.height,
					rect: canvas.getBoundingClientRect()
				});
				console.log("Coordinate system:", calculateCoordinateSystem());
				console.log("Current transform values:", {
					tx_A: getProperty('tx_A'),
					ty_A: getProperty('ty_A'),
					scale_A: getProperty('scale_A', 1),
					angle_deg_A: getProperty('angle_deg_A'),
					tx_B: getProperty('tx_B'),
					ty_B: getProperty('ty_B'),
					scale_B: getProperty('scale_B', 1),
					angle_deg_B: getProperty('angle_deg_B')
				});
				
				// Force refresh of transform data
				await getTransformDataFromNode();
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
				const currentPose = st.which;
				const coords = node.properties ? `${currentPose}: tx=${(getProperty(`tx_${currentPose}`, 0)).toFixed(1)}, ty=${(getProperty(`ty_${currentPose}`, 0)).toFixed(1)}` : "";
				const cacheStatus = st.transformCache.matrices.A ? "Cached" : "No cache";
				const coordSys = calculateCoordinateSystem();
				const aspectRatioInfo = `Aspect: canvas=${(coordSys.actualCanvasWidth/coordSys.actualCanvasHeight).toFixed(2)}, ref=${(coordSys.refW/coordSys.refH).toFixed(2)}`;
				debugInfo.textContent = `Images: ${validCount}/3 | ${previewStatus} | node.imgs: ${imgCount} | ${coords} | ${cacheStatus} | ${aspectRatioInfo}`;
			}, 1000);
			
			node.addDOMWidget("debug_info", "div", debugInfo, {
				serialize: false,
				hideOnZoom: false,
			});

			// Add coordinate system info
			const coordInfo = document.createElement("div");
			coordInfo.style.marginTop = "5px";
			coordInfo.style.padding = "5px";
			coordInfo.style.backgroundColor = "#2a2a2a";
			coordInfo.style.color = "#aaa";
			coordInfo.style.fontSize = "9px";
			coordInfo.style.fontFamily = "monospace";
			coordInfo.style.borderRadius = "3px";
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
			
			node.addDOMWidget("coord_info", "div", coordInfo, {
				serialize: false,
				hideOnZoom: false,
			});
		};
	}
});
