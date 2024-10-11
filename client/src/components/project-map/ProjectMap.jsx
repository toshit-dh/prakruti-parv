/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import animalIndicator from '../../assets/animal-indicator.png';

const customIcon = L.icon({
  iconUrl: animalIndicator, 
  iconSize: [85, 140],
  iconAnchor: [12, 41],
});

const ProjectMap = ({ location }) => {
  const { latitude, longitude } = location;

  return (
    <MapContainer
      center={[latitude, longitude]}
      zoom={13}
      scrollWheelZoom={false}
      style={{ height: '100%', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <Marker position={[latitude, longitude]} icon={customIcon}>
        <Popup>
          Project Location: {latitude}, {longitude}
        </Popup>
      </Marker>
    </MapContainer>
  );
};

export default ProjectMap;