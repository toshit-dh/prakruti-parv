import React, { useEffect, useState } from "react";
import "./MyUploads.css";
import { FaHeart } from "react-icons/fa";
import axios from "axios";
import { GET_UPLOAD_ROUTE } from "../../utils/Routes";

export default function MyUploads() {
  const [activeTab, setActiveTab] = useState("species");
  const [data, setData] = useState({ species: [], poaching: [] });

  useEffect(() => {
    const fetchUploads = async () => {
      try {
        const response = await axios.get(GET_UPLOAD_ROUTE, {
          withCredentials: true,
        });
        const result = response.data.data;
        const uploads = {
          species: result.filter((upload) => upload.category === "species"),
          poaching: result.filter((upload) => upload.category === "poaching"),
        };
        setData(uploads);
      } catch (error) {
        console.error("Error fetching uploads:", error);
      }
    };

    fetchUploads();
  }, []); // Empty dependency array to make sure it runs once

  return (
    <div className="uploads-container">
      <div className="tabs">
        <button
          className={activeTab === "species" ? "active" : ""}
          onClick={() => setActiveTab("species")}
        >
          Species
        </button>
        <button
          className={activeTab === "poaching" ? "active" : ""}
          onClick={() => setActiveTab("poaching")}
        >
          Poaching
        </button>
      </div>
      <div className="posts">
        {data[activeTab].map((post, index) => (
          <div key={index} className="post-card">
            {post.mediaType.includes("video") ? (
              <video controls className="post-media" src={post.mediaUrl}>
                Your browser does not support the video tag.
              </video>
            ) : post.mediaType.includes("audio") ? (
              <audio controls className="post-media" src={post.mediaUrl}>
                Your browser does not support the audio element.
              </audio>
            ) : (
              <img
                src={post.mediaUrl} // Assuming `post.url` is the image URL
                alt={post.title}
                className="post-media"
              />
            )}
            <div className="post-content">
              <img
                className="post-icon"
                src={post.postIconUrl}
                alt="Post Icon"
              />
              <h3 className="post-title" title={post.description}>
                {post.title}
              </h3>
            </div>
            <div className="post-footer">
              {post.isHonoured && (
                <img
                  title="You have been honoured for this post and have received PrakrutiRatna Badge. You have been also credired 5 Prakruti Mudra"
                  className="post-honour-icon"
                  src={post.ratnaUrl}
                  alt="Honour Icon"
                />
              )}
              <span className="likes">
                Likes
                <FaHeart /> {post.likes}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
