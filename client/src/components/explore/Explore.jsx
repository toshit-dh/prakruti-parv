import { useEffect, useState } from "react";
import Navbar from "../navbar/Navbar";
import axios from "axios";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import "./Explore.css";
import tweetsData from './tweets.json';
import postsData from './instaposts.json'


const Explore = () => {
  const [news, setNews] = useState([]);

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8081/fetch_indian_news")
      .then((response) => {
        setNews(response.data.articles || []);
      })
      .catch((error) => console.error("Error fetching news:", error));
  }, []);

  const responsive = {
    desktop: {
      breakpoint: { max: 3000, min: 1024 },
      items: 3,
      partialVisibilityGutter: 40,
    },
    tablet: {
      breakpoint: { max: 1024, min: 464 },
      items: 2,
      partialVisibilityGutter: 30,
    },
    mobile: {
      breakpoint: { max: 464, min: 0 },
      items: 1,
      partialVisibilityGutter: 30,
    },
  };

  return (
    <div className="exploreContainer">
      <Navbar />
      <div className="exploreContent">
        <h1 className="exploreTitle">Wildlife Explore</h1>
        <section className="latestNews">
          <h2>Latest News</h2>
          {news.length > 0 ? (
            <div className="carouselContainer">
              <Carousel
                responsive={responsive}
                containerClass="container-with-dots"
                arrows={false}
                autoPlay
                autoPlaySpeed={3000}
                infinite
                showDots={false}
              >
                {news.map((item, index) => (
                  <div className="newsItem" key={index}>
                    <img
                      src={item.urlToImage || "placeholder.jpg"}
                      alt={item.title}
                      className="newsImage"
                    />
                    <h3 className="newsTitle">{item.title}</h3>
                    <p className="newsDescription">{item.description}</p>
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="newsLink"
                    >
                      Read More
                    </a>
                  </div>
                ))}
              </Carousel>
            </div>
          ) : (
            <p>Loading news...</p>
          )}
        </section>
        <div className="explore-container">
                <section className="tweetsSection">
                <h2>Tweets</h2>
                {tweetsData.tweets.length > 0 ? (
                    <div className="tweetsContainer">
                    {tweetsData.tweets.map((tweet, index) => (
                        <div key={index} className="tweetItem">
                        <iframe 
                            src={tweet.link} 
                            width="100%" 
                            height="300" 
                            frameBorder="0" 
                            title={`Tweet ${index + 1}`} 
                        ></iframe>
                        </div>
                    ))}
                    </div>
                ) : (
                    <p>No tweets available.</p>
                )}
                </section>

                <section className="instaPostsSection">
                     <h2>Instagram Posts</h2>
                     {postsData.length > 0 ? (
                        <div className="instaPostsContainer">
                          {postsData.map((post, index) => (
                            <div key={index} className="instaPostItem" dangerouslySetInnerHTML={{ __html: post.embedCode }}>
                               
                              
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p>No Instagram posts available.</p>
                      )}
                 
                </section>
        </div>
        
      </div>
    </div>
  );
};

export default Explore;
