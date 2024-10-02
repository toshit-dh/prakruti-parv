/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from 'react';
import './PoachingDialog.css';
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
                  {poachInfo.details}
            </div>
            <DetectedFramesCarousel detectedFrames={poachInfo.detected_frames} />
          </div>
        ) : (
          <div className="not-detected">
            <h3>✅ No Poaching Detected</h3>
            <p>{poachInfo.details}</p>
          </div>
        )}
        {poachInfo.poaching_detected ?(
          <button className="closeButton" onClick={onClose}>Close</button>
        ):(
           <button className="closeButton2" onClick={onClose}>Close</button>
        )}
        
      </div>
    </div>
  );
};

export default PoachingDialog;
