import { Typography } from "antd";

function AppFooter() {
  return (
    <div className="AppFooter">
      <Typography.Link href="tel:+123456789" style={{ color: "rgb(255,255,255)"}}  >Project Overview</Typography.Link>
      <Typography.Link href="https://www.google.com" target={"_blank"} style={{ color: "rgb(255,255,255)"}}>
        Privacy Policy
      </Typography.Link>
      <Typography.Link href="https://www.google.com" target={"_blank"} style={{ color: "rgb(255,255,255)"}}>
        About us
      </Typography.Link>
    </div>
  );
}
export default AppFooter;
