import React from 'react';
import './PoachingDialog.css'; // Create a CSS file for styling
import DetectedFramesCarousel from './DetectedFrameCarousel';

const PoachingDialog = ({ isOpen, poachInfo, onClose }) => {
  if (!isOpen) return null; 
  
  return (
    <div className="dialogBackdrop" onClick={onClose}>
      <div className="dialogContainer" onClick={(e) => e.stopPropagation()}>
        {poachInfo.poaching_detected ? (
          <div className="detected">
            <h3>🛑 Poaching Detected 🛑</h3>
            <div className="details">
              <ul>
                <li>Total Frames Processed: {poachInfo.details.total_frames_processed}</li>
                <li>Number of &apos;Yes&apos; Frames: {poachInfo.details.yes_frames}</li>
                <li>Number of &apos;No&apos; Frames: {poachInfo.details.no_frames}</li>
              </ul>
            </div>
            <DetectedFramesCarousel detectedFrames={poachInfo.detected_frames} />
          </div>
        ) : (
          <div className="not-detected">
            <h3>✅ No Poaching Detected</h3>
            <p>{poachInfo.details}</p>
          </div>
        )}
        <button className="closeButton" onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default PoachingDialog;
