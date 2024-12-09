import React, { useEffect, useState } from "react";
import "./Home.css"; // Create a separate CSS file for Home styles

function Home() {
  const [imageIndex, setImageIndex] = useState(0);
  const images = [
    "/assets/outpu-HeartMap.png",
    "/assets/output-Corr-Matricss.png",
    "/assets/Significance.png",
    "/assets/Magnitude.png"
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setImageIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 3000); // Change image every 3 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, [images.length]);

  return (
    <div className="image-container">
      <div className="image-strip" style={{ transform: `translateX(-${imageIndex * 100}%)` }}>
        {images.map((src, index) => (
          <img key={index} src={src} alt={`Slide ${index}`} className="image" />
        ))}
      </div>
    </div>
  );
}

export default Home;
