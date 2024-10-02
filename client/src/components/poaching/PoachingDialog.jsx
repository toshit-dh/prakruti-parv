import React from 'react';
import './PoachingDialog.css'; // Create a CSS file for styling

const PoachingDialog = ({ isOpen, _, onClose }) => {
  if (!isOpen) return null; // Return null if the dialog is not open
  const poachInfo = {
    'poaching_detected': true,
    'details': 'Poaching activities detected in the uploaded video.',
    'detected_frames': ""
         }
  return (
    <div className="dialogBackdrop" onClick={onClose}>
      <div className="dialogContainer" onClick={(e) => e.stopPropagation()}>
        {poachInfo.poaching_detected ? (
          <div className="detected">
            <h3>🛑 Poaching Detected</h3>
            <div className="details">
              <h4>Detection Details:</h4>
              <ul>
                <li>Total Frames Processed: {poachInfo.details.total_frames_processed}</li>
                <li>Number of &apos;Yes&apos; Frames: {poachInfo.details.yes_frames}</li>
                <li>Number of &apos;No&apos; Frames: {poachInfo.details.no_frames}</li>
                {/* <li>Frames with Poaching Detected: {poachInfo.details.yes_frame_indices.join(', ')}</li> */}
              </ul>
            </div>
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
