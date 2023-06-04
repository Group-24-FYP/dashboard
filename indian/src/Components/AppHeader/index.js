import { Image } from "antd";
import { useEffect} from "react";
import logo1 from '../../image/ENTC-vector.png';

function AppHeader() {
  useEffect(() => {
  }, []);

  return (
    <div className="AppHeader">
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <Image width={180} src={logo1} />
        <h1 style={{ color: "rgb(255,255,255)", marginTop: '-5px' }} >Anomaly Detection Dashboard</h1>
      </div>

    </div>
    
      
  );
}
export default AppHeader;
