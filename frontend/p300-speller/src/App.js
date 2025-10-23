import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const P300SpellerSystem = () => {
  // State management
  const [spelledText, setSpelledText] = useState('');
  const [currentWord, setCurrentWord] = useState('');
  const [isSpelling, setIsSpelling] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [eegConnected, setEegConnected] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [flashingCells, setFlashingCells] = useState(new Set());
  const [p300Detection, setP300Detection] = useState('Waiting...');
  const [eegData, setEegData] = useState([]);
  const [metrics, setMetrics] = useState({ accuracy: '--', speed: '--', trials: 0 });
  const [currentMode, setCurrentMode] = useState('simulation');
  
  // Settings
  const [settings, setSettings] = useState({
    flashDuration: 125,
    isiDuration: 125,
    trialsPerChar: 10,
    p300Threshold: 0.7
  });
  
  // WebSocket connection
  const ws = useRef(null);
  const spellingLoop = useRef(null);
  
  // P300 Matrix
  const matrix = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '0']
  ];

  // Connect to backend WebSocket
  const connectWebSocket = useCallback(() => {
    try {
      ws.current = new WebSocket('ws://localhost:8765');
      
      ws.current.onopen = () => {
        setIsConnected(true);
        console.log('Connected to P300 backend');
        // Request system status
        sendMessage({ command: 'get_status' });
      };
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };
      
      ws.current.onclose = () => {
        setIsConnected(false);
        setEegConnected(false);
        console.log('Disconnected from backend');
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
    } catch (error) {
      console.error('Failed to connect to backend:', error);
    }
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'connection_status':
        setIsConnected(data.connected);
        setCurrentMode(data.mode);
        break;
      case 'eeg_status':
        setEegConnected(data.connected);
        setCurrentMode(data.mode);
        break;
      case 'mode_changed':
        setCurrentMode(data.mode);
        alert(`Switched to ${data.mode} mode: ${data.message}`);
        break;
      case 'p300_detection':
        setP300Detection(`P300 detected! (${(data.probability * 100).toFixed(1)}%)`);
        setTimeout(() => setP300Detection('Waiting...'), 2000);
        break;
      case 'eeg_data':
        updateEEGVisualization(data.data, data.p300_prob);
        break;
      case 'session_started':
        console.log('Session started in', data.mode, 'mode');
        break;
      case 'session_stopped':
        console.log('Session stopped');
        break;
      case 'model_test_results':
        alert(`Model Test Results:\nAccuracy: ${(data.results.accuracy * 100).toFixed(1)}%\nPrecision: ${(data.results.precision * 100).toFixed(1)}%\nRecall: ${(data.results.recall * 100).toFixed(1)}%`);
        break;
      default:
        if (data.predictions) {
          setPredictions(data.predictions);
        }
        if (data.session_data) {
          setMetrics({
            accuracy: data.session_data.accuracy ? data.session_data.accuracy.toFixed(1) + '%' : '--',
            speed: calculateSpeed(data.session_data),
            trials: data.session_data.trials || 0
          });
        }
        if (data.error) {
          console.error('Backend error:', data.error);
          alert(`Backend Error: ${data.error}`);
        }
    }
  };

  // Send message to backend
  const sendMessage = (message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
      alert('Not connected to backend. Please refresh the page.');
    }
  };

  // Update EEG visualization
  const updateEEGVisualization = (newData, p300Prob) => {
    if (!newData || !Array.isArray(newData) || newData.length === 0) return;
    
    const timePoints = newData[0].map((_, index) => ({
      time: index * 4, // 4ms per sample at 250Hz
      channel1: newData[0] ? newData[0][index] : 0,
      channel2: newData[1] ? newData[1][index] : 0,
      channel3: newData[2] ? newData[2][index] : 0,
      channel4: newData[3] ? newData[3][index] : 0,
      channel5: newData[4] ? newData[4][index] : 0,
      channel6: newData[5] ? newData[5][index] : 0,
      channel7: newData[6] ? newData[6][index] : 0,
      channel8: newData[7] ? newData[7][index] : 0,
      p300: p300Prob > settings.p300Threshold ? p300Prob * 100 : 0
    }));
    
    setEegData(prevData => {
      const newEegData = [...prevData, ...timePoints].slice(-200); // Keep last 200 points
      return newEegData;
    });
  };

  // Calculate typing speed
  const calculateSpeed = (sessionData) => {
    if (!sessionData.start_time) return '--';
    const elapsed = (new Date() - new Date(sessionData.start_time)) / 1000 / 60; // minutes
    return elapsed > 0 ? Math.round(spelledText.length / elapsed) : 0;
  };

  // Switch between hardware and simulation modes
  const switchMode = (mode) => {
    if (isSpelling) {
      alert('Please stop spelling session before switching modes');
      return;
    }
    sendMessage({ command: 'switch_mode', mode: mode });
  };

  // Start spelling process
  const startSpelling = async () => {
    if (!eegConnected) {
      alert('Please connect EEG hardware/simulation first');
      return;
    }
    
    setIsSpelling(true);
    sendMessage({ command: 'start_spelling' });
    
    // Use a ref to track spelling state for immediate access
    const spellingActive = { current: true };
    
    // Start visual spelling loop
    const runSpellingCycle = async () => {
  while (spellingActive.current) {
    // Checkerboard pattern 1 (like white squares on a chess board)
    const checkerboard1 = [];
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 6; col++) {
        if ((row + col) % 2 === 0) {
          checkerboard1.push(`${row}-${col}`);
        }
      }
    }
    
    setFlashingCells(new Set(checkerboard1));
    sendMessage({ command: 'present_stimulus', type: 'checkerboard', id: 1 });
    await sleep(settings.flashDuration);
    setFlashingCells(new Set());
    await sleep(settings.isiDuration);
    
    // Checkerboard pattern 2 (like black squares on a chess board)
    const checkerboard2 = [];
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 6; col++) {
        if ((row + col) % 2 === 1) {
          checkerboard2.push(`${row}-${col}`);
        }
      }
    }
    
    setFlashingCells(new Set(checkerboard2));
    sendMessage({ command: 'present_stimulus', type: 'checkerboard', id: 2 });
    await sleep(settings.flashDuration);
    setFlashingCells(new Set());
    await sleep(settings.isiDuration);
  }
};
    
    // Store the stop function
    window.stopSpelling = () => { spellingActive.current = false; };
    
    runSpellingCycle();
  };

  // Stop spelling process
  const stopSpelling = () => {
    setIsSpelling(false);
    sendMessage({ command: 'stop_spelling' });
    
    if (window.stopSpelling) {
      window.stopSpelling();
    }
    
    setFlashingCells(new Set());
  };

  // Perform one spelling cycle (flash rows and columns)
  const performSpellingCycle = async () => {
    // Flash rows
    for (let row = 0; row < 6; row++) {
      if (!isSpelling) break;
      
      // Flash row
      const rowCells = Array.from({length: 6}, (_, col) => `${row}-${col}`);
      setFlashingCells(new Set(rowCells));
      
      // Send stimulus presentation to backend
      sendMessage({ 
        command: 'present_stimulus', 
        type: 'row', 
        id: row 
      });
      
      await sleep(settings.flashDuration);
      
      setFlashingCells(new Set());
      await sleep(settings.isiDuration);
    }
    
    // Flash columns
    for (let col = 0; col < 6; col++) {
      if (!isSpelling) break;
      
      // Flash column
      const colCells = Array.from({length: 6}, (row, _) => `${row}-${col}`);
      setFlashingCells(new Set(colCells));
      
      // Send stimulus presentation to backend
      sendMessage({ 
        command: 'present_stimulus', 
        type: 'column', 
        id: col 
      });
      
      await sleep(settings.flashDuration);
      
      setFlashingCells(new Set());
      await sleep(settings.isiDuration);
    }
  };

  // Sleep utility
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  // Connect to EEG hardware
  const connectEEG = () => {
    sendMessage({ command: 'connect_eeg' });
  };

  // Add character to text
  const addCharacter = (char) => {
    setCurrentWord(prev => prev + char);
    setSpelledText(prev => prev + char);
    updatePredictions(spelledText + currentWord + char);
  };

  // Add space
  const addSpace = () => {
    if (currentWord) {
      setSpelledText(prev => prev + ' ');
      setCurrentWord('');
      updatePredictions(spelledText + ' ');
    }
  };

  // Backspace
  const backspace = () => {
    if (currentWord.length > 0) {
      setCurrentWord(prev => prev.slice(0, -1));
      setSpelledText(prev => prev.slice(0, -1));
    } else if (spelledText.length > 0) {
      setSpelledText(prev => prev.slice(0, -1));
    }
    updatePredictions(spelledText.slice(0, -1));
  };

  // Clear all text
  const clearText = () => {
    setSpelledText('');
    setCurrentWord('');
    setPredictions([]);
    setMetrics({ accuracy: '--', speed: '--', trials: 0 });
  };

  // Update predictions
  const updatePredictions = (context) => {
    if (context.trim()) {
      sendMessage({ 
        command: 'get_predictions', 
        context: context 
      });
    } else {
      setPredictions(['Start', 'typing', 'to', 'get', 'suggestions']);
    }
  };

  // Select prediction
  const selectPrediction = (word) => {
    const words = spelledText.split(' ');
    if (words.length > 0 && words[words.length - 1].length > 0) {
      words[words.length - 1] = word;
    } else {
      words.push(word);
    }
    
    const newText = words.join(' ');
    setSpelledText(newText);
    setCurrentWord(word);
    updatePredictions(newText);
  };

  // Test P300 model
  const testModel = () => {
    sendMessage({ command: 'test_model' });
  };

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (spellingLoop.current) {
        clearInterval(spellingLoop.current);
      }
    };
  }, [connectWebSocket]);

  // Update predictions when text changes
  useEffect(() => {
    if (spelledText) {
      updatePredictions(spelledText);
    }
  }, [spelledText]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            P300 Speller System
          </h1>
          <p className="text-lg opacity-80">Brain-Computer Interface with AI-Powered Predictions</p>
        </div>

        {/* Status Bar */}
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-4 mb-6 flex justify-between items-center flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span>Backend: {isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${eegConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span>EEG: {eegConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm opacity-75">Mode:</span>
              <span className={`px-2 py-1 rounded text-xs font-semibold ${
                currentMode === 'hardware' ? 'bg-blue-600' : 'bg-green-600'
              }`}>
                {currentMode.toUpperCase()}
              </span>
            </div>
          </div>
          <div className="text-sm opacity-75">
            Session: {isSpelling ? 'Active' : 'Ready'}
          </div>
        </div>

        {/* Mode Switcher */}
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-4 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-cyan-300">Operating Mode</h3>
          <div className="flex gap-3">
            <button
              onClick={() => switchMode('simulation')}
              disabled={currentMode === 'simulation' || isSpelling}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                currentMode === 'simulation'
                  ? 'bg-green-600 text-white'
                  : 'bg-white bg-opacity-20 hover:bg-opacity-30'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              Simulation Mode
            </button>
            <button
              onClick={() => switchMode('hardware')}
              disabled={currentMode === 'hardware' || isSpelling}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                currentMode === 'hardware'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white bg-opacity-20 hover:bg-opacity-30'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              Hardware Mode
            </button>
          </div>
          <p className="text-sm opacity-75 mt-2">
            {currentMode === 'simulation' 
              ? 'Using simulated EEG data for testing and development'
              : 'Connected to OpenBCI Cyton hardware for real EEG data'
            }
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Speller Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Text Output */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">Spelled Text</h3>
              <div className="bg-black bg-opacity-30 rounded-xl p-4 min-h-[120px] font-mono text-lg relative">
                <span>{spelledText}</span>
                <span className="inline-block w-0.5 h-6 bg-green-400 animate-pulse ml-1"></span>
              </div>
            </div>

            {/* AI Predictions */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">AI Predictions</h3>
              <div className="flex flex-wrap gap-2">
                {predictions.map((word, index) => (
                  <button
                    key={index}
                    onClick={() => selectPrediction(word)}
                    className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 px-3 py-2 rounded-lg transition-all duration-200 hover:scale-105"
                  >
                    {word}
                  </button>
                ))}
              </div>
            </div>

            {/* P300 Matrix */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">P300 Speller Matrix</h3>
              <div className="grid grid-cols-6 gap-3 max-w-md mx-auto">
                {matrix.map((row, rowIndex) =>
                  row.map((char, colIndex) => {
                    const cellKey = `${rowIndex}-${colIndex}`;
                    const isFlashing = flashingCells.has(cellKey);
                    
                    return (
                      <button
                        key={cellKey}
                        className={`
                          aspect-square text-xl font-bold rounded-lg transition-all duration-100
                          ${isFlashing 
                            ? 'bg-gradient-to-br from-yellow-400 to-orange-500 scale-110 shadow-lg' 
                            : 'bg-white bg-opacity-20 hover:bg-opacity-30'
                          }
                        `}
                        onClick={() => addCharacter(char)}
                      >
                        {char}
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Controls */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">Controls</h3>
              <div className="space-y-3">
                <button
                  onClick={connectEEG}
                  disabled={eegConnected}
                  className="w-full bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-3 rounded-lg font-medium transition-all duration-200"
                >
                  {eegConnected ? `EEG Connected (${currentMode})` : 'Connect EEG'}
                </button>
                
                <button
                  onClick={isSpelling ? stopSpelling : startSpelling}
                  disabled={!eegConnected}
                  className={`w-full px-4 py-3 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
                    isSpelling
                      ? 'bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600'
                      : 'bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600'
                  }`}
                >
                  {isSpelling ? 'Stop Spelling' : 'Start Spelling'}
                </button>
                
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={addSpace}
                    className="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-2 rounded-lg transition-all duration-200"
                  >
                    Space
                  </button>
                  <button
                    onClick={backspace}
                    className="bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-2 rounded-lg transition-all duration-200"
                  >
                    Backspace
                  </button>
                </div>
                
                <button
                  onClick={clearText}
                  className="w-full bg-gradient-to-r from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 px-4 py-2 rounded-lg transition-all duration-200"
                >
                  Clear All
                </button>
                
                <button
                  onClick={testModel}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 px-4 py-2 rounded-lg transition-all duration-200"
                >
                  Test P300 Model
                </button>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">Performance</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">{metrics.accuracy}</div>
                  <div className="text-sm opacity-75">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">{metrics.speed}</div>
                  <div className="text-sm opacity-75">Chars/min</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">{metrics.trials}</div>
                  <div className="text-sm opacity-75">Trials</div>
                </div>
              </div>
            </div>

            {/* EEG Visualization */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">EEG Signal</h3>
              <div className="h-40 mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={eegData.slice(-50)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(0,0,0,0.8)', 
                        border: 'none', 
                        borderRadius: '8px',
                        color: 'white'
                      }} 
                    />
                    <Line type="monotone" dataKey="channel1" stroke="#00ff88" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="channel2" stroke="#0088ff" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="p300" stroke="#ff4444" strokeWidth={3} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="text-center">
                <div className="text-sm opacity-75">P300 Detection:</div>
                <div className="text-green-400 font-medium">{p300Detection}</div>
              </div>
            </div>

            {/* Settings */}
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm mb-2">Flash Duration: {settings.flashDuration}ms</label>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    value={settings.flashDuration}
                    onChange={(e) => setSettings(prev => ({...prev, flashDuration: parseInt(e.target.value)}))}
                    className="w-full accent-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-2">ISI Duration: {settings.isiDuration}ms</label>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    value={settings.isiDuration}
                    onChange={(e) => setSettings(prev => ({...prev, isiDuration: parseInt(e.target.value)}))}
                    className="w-full accent-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-2">P300 Threshold: {settings.p300Threshold}</label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    value={settings.p300Threshold}
                    onChange={(e) => setSettings(prev => ({...prev, p300Threshold: parseFloat(e.target.value)}))}
                    className="w-full accent-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default P300SpellerSystem;