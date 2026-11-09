import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  UploadCloud,
  Users,
  AlertTriangle,
  Sliders,
  Eye,
  Download,
  RefreshCw,
  Layers,
  Activity,
  ShieldAlert,
  Sparkles,
  Zap,
  Target,
  FileJson,
  Video,
  VideoOff,
  Camera,
  Play,
  Pause,
  Radio,
} from 'lucide-react';

export default function App() {
  // Input Mode: 'webcam', 'upload', 'samples'
  const [inputMode, setInputMode] = useState('webcam');

  // File Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [viewMode, setViewMode] = useState('annotated');
  const [activeZoneFilter, setActiveZoneFilter] = useState(null);
  const [serverOnline, setServerOnline] = useState(false);

  // High Accuracy Detection Settings
  const [preset, setPreset] = useState('dense');
  const [confidence, setConfidence] = useState(0.15);
  const [iou, setIou] = useState(0.45);
  const [imgsz, setImgsz] = useState(1280);
  const [modelName, setModelName] = useState('yolov8m.pt');
  const [greenCutoff, setGreenCutoff] = useState(10);
  const [redCutoff, setRedCutoff] = useState(25);
  const [drawBoxes, setDrawBoxes] = useState(true);
  const [drawGrid, setDrawGrid] = useState(true);
  const [drawPoints, setDrawPoints] = useState(true);

  // Live Webcam State
  const [cameraActive, setCameraActive] = useState(false);
  const [liveDetectActive, setLiveDetectActive] = useState(true);
  const [streamIntervalMs, setStreamIntervalMs] = useState(200); // 5 FPS
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [fps, setFps] = useState(0);

  // Refs
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(document.createElement('canvas'));
  const mediaStreamRef = useRef(null);
  const isDetectingRef = useRef(false);
  const frameCounterRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());

  // Samples
  const [samples, setSamples] = useState([]);

  // Preset switch handler
  const applyPreset = (mode) => {
    setPreset(mode);
    if (mode === 'dense') {
      setConfidence(0.15);
      setImgsz(1280);
      setModelName('yolov8m.pt');
    } else if (mode === 'balanced') {
      setConfidence(0.25);
      setImgsz(1024);
      setModelName('yolov8s.pt');
    } else if (mode === 'fast') {
      setConfidence(0.30);
      setImgsz(640);
      setModelName('yolov8n.pt');
    }
  };

  // Health check, sample list & media devices enumeration
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then(() => setServerOnline(true))
      .catch(() => setServerOnline(false));

    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.samples) setSamples(data.samples);
      })
      .catch(() => {});

    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((devs) => {
        const videoDevs = devs.filter((d) => d.kind === 'videoinput');
        setDevices(videoDevs);
        if (videoDevs.length > 0) setSelectedDeviceId(videoDevs[0].deviceId);
      }).catch(() => {});
    }

    return () => {
      stopCamera();
    };
  }, []);

  // --------------------------------------------------------------------------
  // Camera Handling
  // --------------------------------------------------------------------------
  const startCamera = async (deviceId = selectedDeviceId) => {
    setErrorMsg(null);
    try {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
      }

      const hasDeviceId = deviceId && typeof deviceId === 'string' && deviceId.trim().length > 0;
      const constraints = {
        video: hasDeviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
      };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (cErr) {
        console.warn('Constrained getUserMedia failed, trying fallback video: true', cErr);
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      mediaStreamRef.current = stream;

      // Unhide wrapper immediately
      setCameraActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch((e) => console.warn('Video play error on metadata:', e));
        };
        videoRef.current.play().catch((e) => console.warn('Direct video play error:', e));
      }

      // Refresh device list with populated labels now that permission is active
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        navigator.mediaDevices.enumerateDevices().then((devs) => {
          const videoDevs = devs.filter((d) => d.kind === 'videoinput');
          if (videoDevs.length > 0) setDevices(videoDevs);
        }).catch(() => {});
      }
    } catch (err) {
      console.error('Camera access failed:', err);
      setErrorMsg(`Webcam error: ${err.message}. Please allow camera access in browser permissions.`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    // Clear overlay canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // --------------------------------------------------------------------------
  // Live Webcam Detection Loop
  // --------------------------------------------------------------------------
  const processLiveFrame = useCallback(async () => {
    if (!videoRef.current || !cameraActive || !liveDetectActive || isDetectingRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2 || video.videoWidth === 0) return;

    isDetectingRef.current = true;
    try {
      const offCanvas = offscreenCanvasRef.current;
      offCanvas.width = video.videoWidth;
      offCanvas.height = video.videoHeight;
      const offCtx = offCanvas.getContext('2d');
      offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

      offCanvas.toBlob(async (blob) => {
        if (!blob) {
          isDetectingRef.current = false;
          return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'webcam_frame.jpg');
        formData.append('confidence', confidence);
        formData.append('iou', iou);
        formData.append('imgsz', imgsz);
        formData.append('model_name', modelName);
        formData.append('green_cutoff', greenCutoff);
        formData.append('red_cutoff', redCutoff);
        formData.append('draw_boxes', 'false'); // Frontend draws boxes smoothly
        formData.append('draw_grid', 'false');
        formData.append('draw_points', 'false');

        try {
          const res = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            setDetectionResult(data);

            // Render live bounding boxes on overlay canvas
            drawOverlay(data, video.videoWidth, video.videoHeight);

            // Calculate FPS
            frameCounterRef.current += 1;
            const now = Date.now();
            if (now - lastFpsTimeRef.current >= 1000) {
              setFps(Math.round((frameCounterRef.current * 1000) / (now - lastFpsTimeRef.current)));
              frameCounterRef.current = 0;
              lastFpsTimeRef.current = now;
            }
          }
        } catch (err) {
          console.warn('Live frame transmission error:', err);
        } finally {
          isDetectingRef.current = false;
        }
      }, 'image/jpeg', 0.85);
    } catch (outerErr) {
      console.warn('Canvas frame capture error:', outerErr);
      isDetectingRef.current = false;
    }
  }, [cameraActive, liveDetectActive, confidence, iou, imgsz, modelName, greenCutoff, redCutoff, drawBoxes, drawGrid, drawPoints]);

  // Streaming timer
  useEffect(() => {
    let timer;
    if (cameraActive && liveDetectActive) {
      timer = setInterval(processLiveFrame, streamIntervalMs);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [cameraActive, liveDetectActive, streamIntervalMs, processLiveFrame]);

  // Draw overlay onto live canvas
  const drawOverlay = (result, videoW, videoH) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (canvas.width !== videoW) canvas.width = videoW;
    if (canvas.height !== videoH) canvas.height = videoH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, videoW, videoH);

    const rowStep = videoH / 3.0;
    const colStep = videoW / 3.0;

    // Draw 3x3 Grid
    if (drawGrid) {
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const x = c * colStep;
          const y = r * rowStep;
          const zid = `Zone_${r}_${c}`;
          const count = result.zone_counts?.[zid] || 0;
          const tier = count >= redCutoff ? 'Red' : count >= greenCutoff ? 'Yellow' : 'Green';

          // Grid boundaries
          ctx.strokeStyle = tier === 'Red' ? 'rgba(239, 68, 68, 0.7)' : tier === 'Yellow' ? 'rgba(245, 158, 11, 0.6)' : 'rgba(16, 185, 129, 0.5)';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, colStep, rowStep);

          // Fill tint
          ctx.fillStyle = tier === 'Red' ? 'rgba(239, 68, 68, 0.15)' : tier === 'Yellow' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.05)';
          ctx.fillRect(x, y, colStep, rowStep);

          // Label
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 16px Inter, sans-serif';
          ctx.fillText(`${zid}: ${count} (${tier})`, x + 12, y + 26);
        }
      }
    }

    // Draw Detections
    result.detections?.forEach((det) => {
      const [x1, y1, x2, y2] = det.box;
      const zid = det.zone_id;
      const count = result.zone_counts?.[zid] || 0;
      const tier = count >= redCutoff ? 'Red' : count >= greenCutoff ? 'Yellow' : 'Green';
      const color = tier === 'Red' ? '#ef4444' : tier === 'Yellow' ? '#f59e0b' : '#10b981';

      if (drawBoxes) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Tag
        ctx.fillStyle = color;
        const tag = `#${det.id} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px Inter, sans-serif';
        const tw = ctx.measureText(tag).width;
        ctx.fillRect(x1, y1 - 18, tw + 8, 18);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tag, x1 + 4, y1 - 4);
      }

      if (drawPoints) {
        const hx = det.center[0];
        const hy = y1 + (y2 - y1) * 0.12;
        ctx.beginPath();
        ctx.arc(hx, hy, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#facc15';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#000000';
        ctx.stroke();
      }
    });
  };

  // Snapshot frame from live webcam
  const captureWebcamSnapshot = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `webcam_snapshot_${Date.now()}.jpg`, { type: 'image/jpeg' });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setInputMode('upload');
        stopCamera();
        runDetection(file);
      }
    }, 'image/jpeg', 0.95);
  };

  // --------------------------------------------------------------------------
  // Static Image File Handling
  // --------------------------------------------------------------------------
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please select a valid image file (.jpg, .png, .jpeg).');
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    runDetection(file);
  };

  const handleSampleSelected = async (sampleName) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const formData = new FormData();
      formData.append('confidence', confidence);
      formData.append('iou', iou);
      formData.append('imgsz', imgsz);
      formData.append('model_name', modelName);
      formData.append('green_cutoff', greenCutoff);
      formData.append('red_cutoff', redCutoff);

      const res = await fetch(`/api/detect/sample/${sampleName}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Sample detection failed');
      const data = await res.json();
      setDetectionResult(data);
      setPreviewUrl(data.annotated_image);
      setViewMode('annotated');
      setSelectedFile(null);
    } catch (err) {
      setErrorMsg(`Failed to analyze sample: ${err.message}.`);
    } finally {
      setLoading(false);
    }
  };

  const runDetection = async (fileToUpload = selectedFile) => {
    if (!fileToUpload) return;
    setLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('confidence', confidence);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('model_name', modelName);
    formData.append('green_cutoff', greenCutoff);
    formData.append('red_cutoff', redCutoff);
    formData.append('draw_boxes', drawBoxes);
    formData.append('draw_grid', drawGrid);
    formData.append('draw_points', drawPoints);

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to detect crowd in image');
      }

      const data = await res.json();
      setDetectionResult(data);
      setViewMode('annotated');
    } catch (err) {
      setErrorMsg(`Detection failed: ${err.message}.`);
    } finally {
      setLoading(false);
    }
  };

  const downloadAnnotatedImage = () => {
    if (!detectionResult?.annotated_image) return;
    const a = document.createElement('a');
    a.href = detectionResult.annotated_image;
    a.download = `crowdpulse_detected_${Date.now()}.jpg`;
    a.click();
  };

  const exportJsonTelemetry = () => {
    if (!detectionResult) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(detectionResult, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `crowdpulse_telemetry_${Date.now()}.json`;
    a.click();
  };

  const filteredDetections = detectionResult?.detections?.filter((d) => {
    if (!activeZoneFilter) return true;
    return d.zone_id === activeZoneFilter;
  }) || [];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <Users size={24} color="#ffffff" />
          </div>
          <div>
            <div className="brand-title">CrowdPulse Live Vision</div>
            <div className="brand-subtitle">
              Live Webcam & Surveillance Crowd Counter with Spatial Multi-Agent Bottleneck Telemetry
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="status-pill">
            <span className={`status-dot ${serverOnline ? 'online' : 'offline'}`}></span>
            {serverOnline ? `Backend Ready (${modelName} @ ${imgsz}px)` : 'Connecting to API (:8000)...'}
          </div>
        </div>
      </header>

      {/* Mode Navigation Tabs */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '18px' }}>
        <button
          className="btn-secondary"
          style={{
            borderColor: inputMode === 'webcam' ? '#38bdf8' : '',
            background: inputMode === 'webcam' ? 'rgba(56, 189, 248, 0.15)' : '',
            padding: '10px 18px',
            fontSize: '0.9rem',
            fontWeight: 600,
          }}
          onClick={() => {
            setInputMode('webcam');
            if (!cameraActive) startCamera();
          }}
        >
          <Radio size={16} color="#ef4444" /> Live Webcam Access
        </button>

        <button
          className="btn-secondary"
          style={{
            borderColor: inputMode === 'upload' ? '#38bdf8' : '',
            background: inputMode === 'upload' ? 'rgba(56, 189, 248, 0.15)' : '',
            padding: '10px 18px',
            fontSize: '0.9rem',
            fontWeight: 600,
          }}
          onClick={() => {
            setInputMode('upload');
            stopCamera();
          }}
        >
          <UploadCloud size={16} color="#38bdf8" /> Upload Image File
        </button>

        <button
          className="btn-secondary"
          style={{
            borderColor: inputMode === 'samples' ? '#38bdf8' : '',
            background: inputMode === 'samples' ? 'rgba(56, 189, 248, 0.15)' : '',
            padding: '10px 18px',
            fontSize: '0.9rem',
            fontWeight: 600,
          }}
          onClick={() => {
            setInputMode('samples');
            stopCamera();
          }}
        >
          <Layers size={16} color="#f59e0b" /> Preloaded Surveillance Feeds
        </button>
      </div>

      {/* Control Panel */}
      <section className="controls-card">
        <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '24px' }}>
          {/* Mode 1: Live Webcam */}
          {inputMode === 'webcam' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Video size={20} color="#38bdf8" />
                  <span style={{ fontWeight: 600, fontSize: '1rem' }}>Camera Stream Controls</span>
                </div>
                {cameraActive && (
                  <span style={{ fontSize: '0.8rem', color: '#10b981', fontWeight: 600 }}>
                    ● Streaming Live {fps > 0 ? `(${fps} FPS)` : ''}
                  </span>
                )}
              </div>

              {devices.length > 1 && (
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Select Camera Device:</label>
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => {
                      setSelectedDeviceId(e.target.value);
                      startCamera(e.target.value);
                    }}
                    style={{
                      width: '100%',
                      background: 'var(--bg-card)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)',
                      padding: '8px',
                      borderRadius: '6px',
                      marginTop: '4px',
                    }}
                  >
                    {devices.map((d, i) => (
                      <option key={d.deviceId} value={d.deviceId}>
                        {d.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {!cameraActive ? (
                  <button className="btn-primary" onClick={() => startCamera()}>
                    <Play size={16} /> Start Webcam
                  </button>
                ) : (
                  <>
                    <button
                      className="btn-secondary"
                      onClick={() => setLiveDetectActive(!liveDetectActive)}
                      style={{ borderColor: liveDetectActive ? '#10b981' : '#f59e0b' }}
                    >
                      {liveDetectActive ? <Pause size={16} /> : <Play size={16} />}
                      {liveDetectActive ? 'Pause AI' : 'Resume AI'}
                    </button>
                    <button className="btn-secondary" onClick={captureWebcamSnapshot}>
                      <Camera size={16} /> Capture Snapshot
                    </button>
                    <button className="btn-secondary" onClick={stopCamera} style={{ color: '#ef4444' }}>
                      <VideoOff size={16} /> Stop Camera
                    </button>
                  </>
                )}
              </div>

              <div className="slider-group" style={{ marginTop: '8px' }}>
                <div className="slider-label">
                  <span>Stream Inference Speed</span>
                  <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
                    {streamIntervalMs} ms ({Math.round(1000 / streamIntervalMs)} FPS)
                  </span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="800"
                  step="50"
                  value={streamIntervalMs}
                  onChange={(e) => setStreamIntervalMs(parseInt(e.target.value))}
                />
              </div>
            </div>
          )}

          {/* Mode 2: Upload File */}
          {inputMode === 'upload' && (
            <div>
              <div
                className={`dropzone ${isDragging ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => e.target.files && handleFileSelected(e.target.files[0])}
                />
                <UploadCloud size={44} className="dropzone-icon" />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '6px' }}>
                  Upload Camera Image or Group Photo
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                  Drag & drop here or <span style={{ color: 'var(--accent-blue)', textDecoration: 'underline' }}>browse</span> (.jpg, .png)
                </p>
                {selectedFile && (
                  <div style={{ marginTop: '12px', fontSize: '0.85rem', color: '#38bdf8', fontWeight: 600 }}>
                    Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Mode 3: Quick Samples */}
          {inputMode === 'samples' && (
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '10px' }}>
                Select Preloaded Camera Feed
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                {samples.map((s) => (
                  <button
                    key={s.id}
                    className="btn-secondary"
                    style={{ padding: '12px', textAlign: 'left', display: 'flex', flexDirection: 'column' }}
                    onClick={() => handleSampleSelected(s.filename)}
                  >
                    <span style={{ fontWeight: 600 }}>{s.filename}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Click to run detection</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Universal Sensitivity & Detection Settings */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600, marginBottom: '6px' }}>
                Accuracy & Density Presets
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    borderColor: preset === 'dense' ? '#38bdf8' : '',
                    background: preset === 'dense' ? 'rgba(56, 189, 248, 0.15)' : '',
                    padding: '8px 6px',
                    fontSize: '0.78rem',
                    justifyContent: 'center',
                  }}
                  onClick={() => applyPreset('dense')}
                >
                  <Sparkles size={14} color="#38bdf8" /> High Precision
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    borderColor: preset === 'balanced' ? '#38bdf8' : '',
                    background: preset === 'balanced' ? 'rgba(56, 189, 248, 0.15)' : '',
                    padding: '8px 6px',
                    fontSize: '0.78rem',
                    justifyContent: 'center',
                  }}
                  onClick={() => applyPreset('balanced')}
                >
                  <Target size={14} color="#f59e0b" /> Balanced
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    borderColor: preset === 'fast' ? '#38bdf8' : '',
                    background: preset === 'fast' ? 'rgba(56, 189, 248, 0.15)' : '',
                    padding: '8px 6px',
                    fontSize: '0.78rem',
                    justifyContent: 'center',
                  }}
                  onClick={() => applyPreset('fast')}
                >
                  <Zap size={14} color="#10b981" /> Real-Time
                </button>
              </div>
            </div>

            <div className="slider-group">
              <div className="slider-label">
                <span>Confidence Threshold</span>
                <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>{(confidence * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.60"
                step="0.05"
                value={confidence}
                onChange={(e) => {
                  setConfidence(parseFloat(e.target.value));
                  setPreset('custom');
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>
                  Model Backbone
                </label>
                <select
                  value={modelName}
                  onChange={(e) => {
                    setModelName(e.target.value);
                    setPreset('custom');
                  }}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    padding: '6px 10px',
                    borderRadius: '6px',
                    fontSize: '0.82rem',
                  }}
                >
                  <option value="yolov8m.pt">YOLOv8 Medium (Most Accurate)</option>
                  <option value="yolov8s.pt">YOLOv8 Small</option>
                  <option value="yolov8n.pt">YOLOv8 Nano (Fastest)</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>
                  Input Resolution
                </label>
                <select
                  value={imgsz}
                  onChange={(e) => {
                    setImgsz(parseInt(e.target.value));
                    setPreset('custom');
                  }}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    padding: '6px 10px',
                    borderRadius: '6px',
                    fontSize: '0.82rem',
                  }}
                >
                  <option value={1280}>1280 px (HD - Best for Crowds)</option>
                  <option value={1024}>1024 px</option>
                  <option value={640}>640 px (High FPS)</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px', marginTop: '2px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', cursor: 'pointer' }}>
                <input type="checkbox" checked={drawBoxes} onChange={(e) => setDrawBoxes(e.target.checked)} />
                Bounding Boxes
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', cursor: 'pointer' }}>
                <input type="checkbox" checked={drawGrid} onChange={(e) => setDrawGrid(e.target.checked)} />
                3x3 Grid Zones
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', cursor: 'pointer' }}>
                <input type="checkbox" checked={drawPoints} onChange={(e) => setDrawPoints(e.target.checked)} />
                Head Points
              </label>
            </div>

            {inputMode !== 'webcam' && (
              <button
                className="btn-primary"
                style={{ marginTop: 'auto' }}
                onClick={() => runDetection()}
                disabled={loading || (!selectedFile && !detectionResult)}
              >
                {loading ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" /> Detecting Crowd...
                  </>
                ) : (
                  <>
                    <Users size={18} /> Re-Run Live Detection
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </section>

      {/* Error Alert */}
      {errorMsg && (
        <div className="alert-banner CRITICAL">
          <AlertTriangle size={20} />
          <div>{errorMsg}</div>
        </div>
      )}

      {/* Real-Time Telemetry Metrics */}
      {detectionResult && (
        <>
          {detectionResult.alert && (
            <div className={`alert-banner ${detectionResult.alert.status_level}`}>
              <ShieldAlert size={22} />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700 }}>
                  SYSTEM STATUS: {detectionResult.alert.system_status}
                </div>
                {detectionResult.alert.rerouting_directives?.length > 0 ? (
                  <div style={{ marginTop: '4px', fontSize: '0.88rem' }}>
                    {detectionResult.alert.rerouting_directives.map((dir, idx) => (
                      <div key={idx}>• {dir}</div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>
                    All spatial zones operating within safe pedestrian flow limits.
                  </div>
                )}
              </div>
            </div>
          )}

          <section className="metrics-grid">
            <div className="metric-card green">
              <span className="metric-label">Total Detected Crowd</span>
              <div className="metric-value">
                {detectionResult.total_count}
                <span className="metric-unit">people</span>
              </div>
            </div>

            <div className="metric-card red">
              <span className="metric-label">Critical Bottlenecks (Red)</span>
              <div className="metric-value" style={{ color: detectionResult.alert.red_zones?.length > 0 ? '#ef4444' : '#10b981' }}>
                {detectionResult.alert.red_zones?.length || 0}
                <span className="metric-unit">zones</span>
              </div>
            </div>

            <div className="metric-card yellow">
              <span className="metric-label">Elevated Density (Yellow)</span>
              <div className="metric-value" style={{ color: detectionResult.alert.yellow_zones?.length > 0 ? '#f59e0b' : '#94a3b8' }}>
                {detectionResult.alert.yellow_zones?.length || 0}
                <span className="metric-unit">zones</span>
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Inference Latency</span>
              <div className="metric-value" style={{ color: '#38bdf8' }}>
                {detectionResult.inference_time_ms}
                <span className="metric-unit">ms</span>
              </div>
            </div>
          </section>
        </>
      )}

      {/* Main Visual Stream + 3x3 Heatmap Grid */}
      <div className="dashboard-grid">
        {/* Visual Stream Card */}
        <div className="viewer-card">
          <div className="viewer-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Eye size={18} color="var(--accent-blue)" />
              <span style={{ fontWeight: 600 }}>
                {inputMode === 'webcam' ? 'Live Camera Feed with AI Detection' : 'Surveillance Image Stream'}
              </span>
            </div>

            {inputMode === 'webcam' ? (
              <div style={{ display: 'flex', gap: '8px' }}>
                {cameraActive && (
                  <button className="btn-secondary" onClick={captureWebcamSnapshot}>
                    <Camera size={15} /> Snapshot
                  </button>
                )}
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '8px' }}>
                {detectionResult && (
                  <>
                    <button
                      className={`btn-secondary ${viewMode === 'annotated' ? 'active' : ''}`}
                      onClick={() => setViewMode('annotated')}
                      style={{ borderColor: viewMode === 'annotated' ? 'var(--accent-blue)' : '' }}
                    >
                      Annotated
                    </button>
                    {selectedFile && (
                      <button
                        className={`btn-secondary ${viewMode === 'original' ? 'active' : ''}`}
                        onClick={() => setViewMode('original')}
                        style={{ borderColor: viewMode === 'original' ? 'var(--accent-blue)' : '' }}
                      >
                        Original
                      </button>
                    )}
                    <button className="btn-secondary" onClick={downloadAnnotatedImage} title="Save Annotated Image">
                      <Download size={15} /> Save
                    </button>
                    <button className="btn-secondary" onClick={exportJsonTelemetry} title="Export JSON">
                      <FileJson size={15} /> JSON
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Video or Image Canvas */}
          <div className="viewer-canvas-wrapper">
            {inputMode === 'webcam' ? (
              <div className="webcam-container">
                {/* Live video and canvas overlay are always mounted in DOM */}
                <div
                  className="webcam-video-wrapper"
                  style={{ display: cameraActive ? 'inline-flex' : 'none' }}
                >
                  <div className="live-badge">● LIVE CAM</div>
                  {fps > 0 && <div className="fps-badge">{fps} FPS</div>}
                  <video
                    ref={videoRef}
                    className="webcam-video"
                    playsInline
                    muted
                    autoPlay
                  />
                  <canvas
                    ref={canvasRef}
                    className="webcam-overlay-canvas"
                  />
                </div>

                {!cameraActive && (
                  <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '60px 20px' }}>
                    <Video size={56} style={{ opacity: 0.5, marginBottom: '16px', color: 'var(--accent-blue)' }} />
                    <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '8px' }}>
                      Live Camera Feed Inactive
                    </div>
                    <div style={{ fontSize: '0.88rem', color: '#94a3b8', maxWidth: '380px', margin: '0 auto 18px' }}>
                      Click the "Start Webcam" button in the control panel to initialize real-time crowd detection through your camera.
                    </div>
                    <button className="btn-primary" onClick={() => startCamera()} style={{ margin: '0 auto' }}>
                      <Play size={16} /> Start Webcam Now
                    </button>
                  </div>
                )}
              </div>
            ) : detectionResult ? (
              <img
                src={viewMode === 'annotated' ? detectionResult.annotated_image : previewUrl}
                alt="Surveillance Feed"
                className="viewer-img"
              />
            ) : (
              <div style={{ color: 'var(--text-secondary)' }}>
                Upload an image or switch to Live Webcam mode above.
              </div>
            )}
          </div>
        </div>

        {/* Spatial 3x3 Heatmap Grid Card */}
        <div className="viewer-card">
          <div className="viewer-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="var(--accent-cyan)" />
              <span style={{ fontWeight: 600 }}>3x3 Spatial Grid Heatmap</span>
            </div>
            {activeZoneFilter && (
              <button
                className="btn-secondary"
                style={{ fontSize: '0.75rem', padding: '3px 8px' }}
                onClick={() => setActiveZoneFilter(null)}
              >
                Clear Filter
              </button>
            )}
          </div>

          <div className="heatmap-grid">
            {detectionResult?.zones_summary?.map((zone) => {
              const isFiltered = activeZoneFilter === zone.zone_id;
              return (
                <div
                  key={zone.zone_id}
                  className={`heatmap-cell ${zone.tier}`}
                  style={{ outline: isFiltered ? '2px solid #38bdf8' : 'none' }}
                  onClick={() => setActiveZoneFilter(isFiltered ? null : zone.zone_id)}
                  title={`Click to filter detections in ${zone.zone_id}`}
                >
                  <div className="heatmap-cell-id">{zone.zone_id}</div>
                  <div className="heatmap-cell-count">{zone.count}</div>
                  <div className="heatmap-cell-tier">{zone.tier}</div>
                </div>
              );
            })}
            {!detectionResult && (
              <div style={{ gridColumn: 'span 3', textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
                Waiting for detection telemetry...
              </div>
            )}
          </div>

          {/* Zone Telemetry Table */}
          {detectionResult && (
            <div style={{ marginTop: '16px' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Zone Agent Registry
              </div>
              <div className="data-table-wrapper" style={{ maxHeight: '220px' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Zone</th>
                      <th>Occupancy</th>
                      <th>Tier</th>
                      <th>Alert Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detectionResult.zones_summary?.map((z) => (
                      <tr key={z.zone_id} style={{ cursor: 'pointer' }} onClick={() => setActiveZoneFilter(z.zone_id)}>
                        <td style={{ fontWeight: 600 }}>{z.zone_id}</td>
                        <td>{z.count} heads</td>
                        <td>
                          <span style={{ color: z.tier === 'Red' ? '#ef4444' : z.tier === 'Yellow' ? '#f59e0b' : '#10b981', fontWeight: 700 }}>
                            {z.tier}
                          </span>
                        </td>
                        <td>{z.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Detections Log Table */}
      {detectionResult && (
        <div className="viewer-card" style={{ marginTop: '10px' }}>
          <div className="viewer-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} color="var(--accent-blue)" />
              <span style={{ fontWeight: 600 }}>
                Detected Individuals Telemetry ({filteredDetections.length} of {detectionResult.total_count} shown)
              </span>
            </div>
          </div>

          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th># ID</th>
                  <th>Zone</th>
                  <th>Confidence</th>
                  <th>Bounding Box [X1, Y1, X2, Y2]</th>
                  <th>Center (X, Y)</th>
                </tr>
              </thead>
              <tbody>
                {filteredDetections.map((d) => (
                  <tr key={d.id}>
                    <td style={{ fontWeight: 600 }}>Person #{d.id}</td>
                    <td>
                      <span style={{ padding: '2px 8px', borderRadius: '4px', background: 'var(--bg-card)', fontSize: '0.8rem' }}>
                        {d.zone_id}
                      </span>
                    </td>
                    <td style={{ color: '#38bdf8', fontWeight: 600 }}>
                      {(d.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      [{d.box.join(', ')}]
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      ({d.center.join(', ')})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
