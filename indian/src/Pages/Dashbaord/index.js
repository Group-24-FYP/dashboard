import { Card, Space } from "antd";
import React, { useState } from "react";
import img1 from '../../image/ASM.png';
// import video1 from '../../video/toothbrush_conveyorbelt_R1.mp4'// H264 MPEG-4 AVC
//import video1 from '../../video/20230601_140508_converted_vlc.mp4'
import video1 from "../../video/c_20230601_140508.mp4"
import ReviewsBar from "../../reviews/ReviewsBar";
import statusText from '../../Model_data/status.txt';
import scoreText from '../../Model_data/score.txt';
// import Webcam from "react-webcam";
 
function Dashboard() {
  const [status, setStatus] = useState('');
  const [score, setScore] = useState('');
  
    fetch(statusText)
    .then(r => r.text())
    .then(text => {
      setStatus(text);;
    });

    fetch(scoreText)
    .then(r => r.text())
    .then(text => {
      setScore(text);;
    });

  return (

    <div>
    <Space size={50} direction="horizontal" >
 
        {/* LEFT */}
      <div style={{ textAlign: 'center', 
              width: 1100,
              height: 800,
              color: "rgb(255,255,255)",
              backgroundColor: "rgb(12, 19, 79)", 
              fontSize: 24,}}>
        <Space size={10} direction="vertical" align="baseline">
            <div style={{ marginTop: '50px' }}>
            <h1>Annotated Video Stream</h1>
              <div style={{ marginTop: '-180px' }}>
                <video id="myVideo" controls autoPlay width="600" style={{ marginBottom: '-180px', transform: 'rotate(90deg)' }}>
                  <source src={video1} type="video/mp4" />
                </video>
              </div>
  
            </div>
              
            </Space>
      </div>


        {/* RIGHT */}
      <div style={{ textAlign: 'center', 
              width: 650,
              height: 800,
              color: "rgb(255,255,255)",
              backgroundColor: "rgb(12, 19, 79)",
              borderRadius: 20,
              fontSize: 24,
              padding: 20,
              margin: "0 auto",  
              }}>
      
        <Card style={{
                height: 250,
                color: "rgb(255,255,255)",
                backgroundColor: "rgba(92, 70, 156,0.25)",
                borderRadius: 50,
                fontSize: 24,
                padding: 0,
                width: 400,
                margin: "0 auto", 
              }}>
              <Space size={10} direction="vertical" align="baseline">
                  <h2>Uncertainty Score</h2>
                  <div  style={{
                        width: 150,
                        height: 150
                      }}>
                        
                      <ReviewsBar  style={{ marginLeft: "100px" }} score= {score}  />
                  </div>
                </Space>
        </Card>

        <Card style={{
                width: '100%',
                height: '38%',
                color: "rgb(255,255,255)",
                backgroundColor: "rgba(92, 70, 156,0.25)",
                borderRadius: 50,
                fontSize: 24,
                padding: 0,
                marginTop: "20px"
              }}>
              <Space size={10} direction="vertical" align="baseline">
                  <div style={{ marginTop: '-160px' }}>
                    <h2>Anomaly Segmentation Map</h2>
                    <img src={img1} alt="" width="400" height="200" />
                  </div>
                </Space>
        </Card>

        <Card style={{
                color: "rgb(255,255,255)",
                backgroundColor: "rgb(29, 38, 125)",
                borderRadius: 50,
                fontSize: 24,
                padding: 0,
                width: 400,
                height: 200,
                margin: "0 auto", 
                marginTop: "20px"
              }}>
                <div style={{ marginTop: '-20px' }}>
                  <h3>Status</h3>
                  <h4>{status}</h4>

                </div>
        </Card>

      </div>

    </Space>
    </div>
    
  );
}

export default Dashboard;