/* pose_align_widget_optimized.js – Compact canvas widget with full functionality */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "AInseven.PoseAlignCanvasWidget.Optimized",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "PoseAlignTwoToOne") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			const node = this;

			// Create fixed-size canvas
			const CANVAS_SIZE = 512;
			const canvas = document.createElement("canvas");
			Object.assign(canvas, { width: CANVAS_SIZE, height: CANVAS_SIZE });
			Object.assign(canvas.style, {
				border: "2px solid #555", backgroundColor: "#1a1a1a", display: "block",
				cursor: "crosshair", borderRadius: "4px", width: `${CANVAS_SIZE}px`,
				height: `${CANVAS_SIZE}px`, objectFit: "contain", flexShrink: "0"
			});
			const ctx = canvas.getContext("2d");

			// Compact state management
			const st = { 
				dragging: false, which: "A", lastX: 0, lastY: 0, hovering: false,
				loadedImages: { ref: null, A: null, B: null }, previewImage: null,
				refImageSize: { width: 512, height: 512 }, lastProperties: {},
				transformCache: { lastUpdate: 0, matrices: { A: null, B: null }, offsetCorrections: { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } } }
			};

			// Utility functions
			const getProperty = (key, def = 0) => node.properties?.[key] ?? def;
			const setVal = (key, v) => {
				if (key.includes('angle_deg_')) v = ((v % 360) + 360) % 360;
				if (node.properties[key] !== v) {
					node.setProperty(key, v);
					const w = node.widgets.find(w => w.name === key);
					if (w) { w.value = v; w.callback?.(v); }
				}
			};

			const buildAffineMatrix = (scale, angleDeg, tx, ty, cx, cy) => {
				const angleRad = angleDeg * Math.PI / 180;
				const [cosA, sinA] = [Math.cos(angleRad), Math.sin(angleRad)];
				const [R11, R12, R21, R22] = [cosA * scale, -sinA * scale, sinA * scale, cosA * scale];
				return {
					a: R11, b: R21, c: R12, d: R22,
					e: tx + cx - (R11 * cx + R12 * cy),
					f: ty + cy - (R21 * cx + R22 * cy)
				};
			};

			const calculateCoordinateSystem = () => {
				const { width: actualCanvasWidth, height: actualCanvasHeight } = canvas;
				const { width: refW, height: refH } = st.refImageSize;
				const canvasAspect = actualCanvasWidth / actualCanvasHeight;
				const refAspect = refW / refH;
				
				let canvasScale, offsetX, offsetY;
				if (refAspect > canvasAspect) {
					canvasScale = actualCanvasWidth / refW * 0.9;
					offsetX = actualCanvasWidth * 0.05;
					offsetY = (actualCanvasHeight - refH * canvasScale) / 2;
				} else {
					canvasScale = actualCanvasHeight / refH * 0.9;
					offsetY = actualCanvasHeight * 0.05;
					offsetX = (actualCanvasWidth - refW * canvasScale) / 2;
				}
				return { canvasScale, offsetX, offsetY, refW, refH, actualCanvasWidth, actualCanvasHeight };
			};

			const getCanvasCoordinates = e => {
				const rect = canvas.getBoundingClientRect();
				return {
					x: (e.clientX - rect.left) * (canvas.width / rect.width),
					y: (e.clientY - rect.top) * (canvas.height / rect.height)
				};
			};

			// Transform data management
			const getTransformDataFromNode = async () => {
				try {
					const response = await api.fetchApi(`/AInseven/pose_align_data/${node.id}`);
					if (response.ok) {
						const data = await response.json();
						if (data.timestamp > st.transformCache.lastUpdate) {
							st.transformCache = {
								lastUpdate: data.timestamp,
								matrices: data.matrices || { A: null, B: null },
								offsetCorrections: data.offsetCorrections || { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } }
							};
							return true;
						}
					}
				} catch (error) {
					console.log("No transform data available from node");
				}
				return false;
			};

			const checkPropertiesChanged = () => {
				const currentProps = {
					tx_A: getProperty('tx_A'), ty_A: getProperty('ty_A'), scale_A: getProperty('scale_A', 1), angle_deg_A: getProperty('angle_deg_A'),
					tx_B: getProperty('tx_B'), ty_B: getProperty('ty_B'), scale_B: getProperty('scale_B', 1), angle_deg_B: getProperty('angle_deg_B')
				};
				const changed = Object.keys(currentProps).some(key => st.lastProperties[key] !== currentProps[key]);
				if (changed) st.lastProperties = { ...currentProps };
				return changed;
			};

			// Image processing
			const extractPosesFromPreview = async (previewImage) => {
				if (!previewImage) return { ref: null, A: null, B: null };
				const width = previewImage.width / 3, height = previewImage.height;
				st.refImageSize = { width, height };
				
				const poses = {};
				for (let i = 0; i < 3; i++) {
					const tempCanvas = document.createElement('canvas');
					Object.assign(tempCanvas, { width, height });
					const tempCtx = tempCanvas.getContext('2d');
					tempCtx.drawImage(previewImage, i * width, 0, width, height, 0, 0, width, height);
					poses[['ref', 'A', 'B'][i]] = await createImageBitmap(tempCanvas);
				}
				return poses;
			};

			const getImagesFromNode = async () => {
				try {
					if (node.imgs?.[0]?.src) {
						const img = new Image();
						img.src = node.imgs[0].src;
						await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; });
						st.previewImage = img;
						return await extractPosesFromPreview(img);
					}
				} catch (error) {
					console.error("Error getting images from node:", error);
				}
				return { ref: null, A: null, B: null };
			};

			const generatePoseVisualization = (type, width = 384, height = 384) => {
				const canvas = document.createElement('canvas');
				Object.assign(canvas, { width, height });
				const ctx = canvas.getContext('2d');
				
				ctx.fillStyle = '#1a1a1a';
				ctx.fillRect(0, 0, width, height);
				
				const colors = { ref: '#666', A: '#ff4a4a', B: '#4a9eff' };
				ctx.strokeStyle = colors[type];
				ctx.lineWidth = 3;
				ctx.lineCap = 'round';
				
				const [centerX, centerY, scale] = [width / 2, height / 2, 50];
				
				// Simple stick figure
				ctx.beginPath(); ctx.arc(centerX, centerY - scale, 15, 0, Math.PI * 2); ctx.stroke();
				ctx.beginPath(); ctx.moveTo(centerX, centerY - scale + 15); ctx.lineTo(centerX, centerY + scale); ctx.stroke();
				ctx.beginPath(); ctx.moveTo(centerX - scale * 0.8, centerY - scale * 0.3); ctx.lineTo(centerX + scale * 0.8, centerY - scale * 0.3); ctx.stroke();
				ctx.beginPath(); ctx.moveTo(centerX, centerY + scale); ctx.lineTo(centerX - scale * 0.5, centerY + scale * 1.5); 
				ctx.moveTo(centerX, centerY + scale); ctx.lineTo(centerX + scale * 0.5, centerY + scale * 1.5); ctx.stroke();
				
				ctx.fillStyle = ctx.strokeStyle; ctx.font = '12px monospace'; ctx.textAlign = 'center';
				ctx.fillText(`${type.toUpperCase()} (placeholder)`, centerX, height - 20);
				return canvas;
			};

			// Main drawing function
			const draw = async () => {
				try {
					const dataUpdated = await getTransformDataFromNode();
					const images = await getImagesFromNode();
					st.loadedImages = images;
					const hasValidImages = !!(images.ref || images.A || images.B);
					const coordSys = calculateCoordinateSystem();
					const { canvasScale, offsetX, offsetY, refW, refH, actualCanvasWidth, actualCanvasHeight } = coordSys;
					
					// Clear and setup
					ctx.setTransform(1, 0, 0, 1, 0, 0);
					ctx.clearRect(0, 0, actualCanvasWidth, actualCanvasHeight);
					ctx.fillStyle = "#1a1a1a";
					ctx.fillRect(0, 0, actualCanvasWidth, actualCanvasHeight);
					
					// Draw grid
					ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
					const gridStep = 32 * canvasScale;
					for (let i = offsetX; i < actualCanvasWidth; i += gridStep) {
						ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, actualCanvasHeight); ctx.stroke();
					}
					for (let i = offsetY; i < actualCanvasHeight; i += gridStep) {
						ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(actualCanvasWidth, i); ctx.stroke();
					}
					ctx.setLineDash([]);

					// Draw pose layers
					const drawPoseLayer = (img, keyPrefix, color, isActive = false) => {
						if (!img && hasValidImages) return;
						if (!img) img = generatePoseVisualization(keyPrefix === "REF" ? "ref" : keyPrefix.toLowerCase(), refW, refH);

						ctx.setTransform(1, 0, 0, 1, 0, 0);

						if (keyPrefix === "REF") {
							ctx.globalAlpha = 0.5;
							ctx.drawImage(img, offsetX, offsetY, refW * canvasScale, refH * canvasScale);
						} else {
							let tx, ty, sc, rotD;
							const cx = refW / 2.0, cy = refH / 2.0;

							if (st.transformCache.matrices[keyPrefix]) {
								const matrix = st.transformCache.matrices[keyPrefix];
								sc = Math.sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1]);
								rotD = Math.atan2(matrix[1], matrix[0]) * 180 / Math.PI;
								tx = matrix[4]; ty = matrix[5];
							} else {
								tx = getProperty(`tx_${keyPrefix}`, 0); ty = getProperty(`ty_${keyPrefix}`, 0);
								sc = getProperty(`scale_${keyPrefix}`, 1); rotD = getProperty(`angle_deg_${keyPrefix}`, 0);
								const offset = st.transformCache.offsetCorrections[keyPrefix];
								if (offset) { tx += offset.x; ty += offset.y; }
							}

							const affineMatrix = buildAffineMatrix(sc, rotD, tx, ty, cx, cy);
							ctx.setTransform(
								affineMatrix.a * canvasScale, affineMatrix.b * canvasScale,
								affineMatrix.c * canvasScale, affineMatrix.d * canvasScale,
								affineMatrix.e * canvasScale + offsetX, affineMatrix.f * canvasScale + offsetY
							);

							ctx.globalAlpha = 0.9;
							ctx.drawImage(img, 0, 0, refW, refH);
							
							ctx.setTransform(1, 0, 0, 1, 0, 0);
							const corners = [[0, 0], [refW, 0], [refW, refH], [0, refH]].map(([x, y]) => [
								(affineMatrix.a * x + affineMatrix.c * y + affineMatrix.e) * canvasScale + offsetX,
								(affineMatrix.b * x + affineMatrix.d * y + affineMatrix.f) * canvasScale + offsetY
							]);

							ctx.globalAlpha = isActive ? 1.0 : 0.6;
							ctx.strokeStyle = color; ctx.lineWidth = isActive ? 3 : 2;
							ctx.setLineDash(isActive ? [] : [5, 5]);
							ctx.beginPath(); ctx.moveTo(corners[0][0], corners[0][1]);
							corners.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
							ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
						}
					};
					
					drawPoseLayer(images.ref, "REF", "#666");
					drawPoseLayer(images.B, "B", "#4a9eff", st.which === "B");
					drawPoseLayer(images.A, "A", "#ff4a4a", st.which === "A");
					
					// UI elements
					ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1.0;
					
					if (!hasValidImages) {
						ctx.fillStyle = "#888"; ctx.font = "14px monospace"; ctx.textAlign = "center";
						ctx.fillText("Run workflow to generate images", actualCanvasWidth/2, actualCanvasHeight/2 - 20);
					}
					
					// Active pose indicator
					ctx.fillStyle = st.which === "A" ? "#ff4a4a" : "#4a9eff";
					ctx.fillRect(10, 10, 20, 20); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(10, 10, 20, 20);
					ctx.fillStyle = "#fff"; ctx.font = "12px monospace"; ctx.textAlign = "left";
					ctx.fillText(`Active: Pose ${st.which}`, 40, 25);
					
					// Instructions
					ctx.fillStyle = "#aaa"; ctx.font = "10px monospace";
					["Left-click: Move Pose A", "Right-click: Move Pose B", "Wheel: Scale | Shift+Wheel: Rotate", 
					 "Arrow keys: Fine movement | R: Reset pose"].forEach((text, i) => 
						ctx.fillText(text, 10, actualCanvasHeight - 55 + i * 15));
					
				} catch (error) {
					console.error("Error in draw function:", error);
					ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
					ctx.fillStyle = "#1a1a1a"; ctx.fillRect(0, 0, canvas.width, canvas.height);
					ctx.fillStyle = "#ff4444"; ctx.font = "14px monospace"; ctx.textAlign = "center";
					ctx.fillText("Error loading images", canvas.width/2, canvas.height/2);
				}
			};

			// Event handlers
			["contextmenu"].forEach(event => canvas.addEventListener(event, e => e.preventDefault()));
			
			canvas.addEventListener("mousedown", e => {
				st.dragging = true; st.which = e.button === 2 ? "B" : "A";
				const coords = getCanvasCoordinates(e);
				st.lastX = coords.x; st.lastY = coords.y;
				canvas.style.cursor = "grabbing"; draw();
			});
			
			["mouseup", "mouseleave"].forEach(event => 
				canvas.addEventListener(event, () => { st.dragging = false; canvas.style.cursor = "crosshair"; }));
			
			canvas.addEventListener("mouseenter", () => st.hovering = true);
			
			canvas.addEventListener("mousemove", e => {
				if (!st.dragging) return;
				const coords = getCanvasCoordinates(e);
				const [dx, dy] = [coords.x - st.lastX, coords.y - st.lastY];
				st.lastX = coords.x; st.lastY = coords.y;
				
				const { canvasScale } = calculateCoordinateSystem();
				const [scaledDx, scaledDy] = [dx / canvasScale, dy / canvasScale];
				const p = st.which;
				setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) + scaledDx);
				setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) + scaledDy);
				draw();
			});
			
			canvas.addEventListener("wheel", e => {
				e.preventDefault();
				const p = st.which;
				if (e.shiftKey) {
					const rotationStep = e.deltaY > 0 ? 5 : -5;
					setVal(`angle_deg_${p}`, getProperty(`angle_deg_${p}`, 0) + rotationStep);
				} else {
					const cur = getProperty(`scale_${p}`, 1);
					const scaleStep = e.deltaY > 0 ? -0.05 : 0.05;
					setVal(`scale_${p}`, Math.max(0.1, cur + scaleStep));
				}
				draw();
			});

			canvas.addEventListener("keydown", e => {
				if (!st.hovering) return;
				const { canvasScale } = calculateCoordinateSystem();
				const step = (e.shiftKey ? 10 : 1) / canvasScale;
				const p = st.which;
				
				const keyActions = {
					'ArrowLeft': () => setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) - step),
					'ArrowRight': () => setVal(`tx_${p}`, getProperty(`tx_${p}`, 0) + step),
					'ArrowUp': () => setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) - step),
					'ArrowDown': () => setVal(`ty_${p}`, getProperty(`ty_${p}`, 0) + step),
					'a': () => st.which = "A", 'A': () => st.which = "A",
					'b': () => st.which = "B", 'B': () => st.which = "B",
					'r': () => ['tx', 'ty', 'scale', 'angle_deg'].forEach(param => 
						setVal(`${param}_${p}`, param === 'scale' ? 1.0 : 0)),
					'R': () => ['tx', 'ty', 'scale', 'angle_deg'].forEach(param => 
						setVal(`${param}_${p}`, param === 'scale' ? 1.0 : 0))
				};
				
				if (keyActions[e.key]) { e.preventDefault(); keyActions[e.key](); draw(); }
			});
			
			canvas.tabIndex = 0;

			// Widget integration
			const canvasContainer = document.createElement("div");
			Object.assign(canvasContainer.style, {
				display: "flex", justifyContent: "center", alignItems: "center",
				width: "100%", minHeight: `${CANVAS_SIZE}px`, overflow: "visible"
			});
			canvasContainer.appendChild(canvas);
			node.addDOMWidget("pose_canvas", "div", canvasContainer, { serialize: false, hideOnZoom: false });

			// Monitoring and updates
			const monitoredParams = ['tx_A', 'ty_A', 'scale_A', 'angle_deg_A', 'tx_B', 'ty_B', 'scale_B', 'angle_deg_B'];
			
			const setupWidgetMonitoring = () => {
				monitoredParams.forEach(paramName => {
					const widget = node.widgets?.find(w => w.name === paramName);
					if (widget) {
						const originalCallback = widget.callback;
						widget.callback = function(value) {
							originalCallback?.call(this, value);
							setTimeout(draw, 10);
						};
					}
				});
			};
			setTimeout(setupWidgetMonitoring, 100);

			let propertyMonitorInterval = setInterval(async () => {
				const dataUpdated = await getTransformDataFromNode();
				const propsChanged = checkPropertiesChanged();
				if (dataUpdated || propsChanged) draw();
			}, 500);

			// Override node methods
			const originalOnExecuted = node.onExecuted;
			node.onExecuted = function(message) {
				originalOnExecuted?.apply(this, arguments);
				if (message?.images?.length > 0) setTimeout(draw, 100);
				setTimeout(async () => {
					if (await getTransformDataFromNode() || checkPropertiesChanged()) draw();
				}, 1000);
			};

			const originalOnPropertyChanged = node.onPropertyChanged;
			node.onPropertyChanged = function(name, value) {
				originalOnPropertyChanged?.apply(this, arguments);
				if (monitoredParams.includes(name)) draw();
			};

			const handleResize = () => setTimeout(draw, 100);
			window.addEventListener('resize', handleResize);
			
			// Cleanup
			const originalOnRemoved = node.onRemoved;
			node.onRemoved = function() {
				clearInterval(propertyMonitorInterval);
				window.removeEventListener('resize', handleResize);
				originalOnRemoved?.apply(this, arguments);
			};

			setTimeout(draw, 500);
		};
	}
});
