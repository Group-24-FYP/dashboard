import "./App.css";
import AppFooter from "./Components/AppFooter";
import AppHeader from "./Components/AppHeader";
import PageContent from "./Components/PageContent";

function App() {
  return (
    <div className="App">
      <AppHeader />
      <div className="SideMenuAndPageContent">
        {/* <SideMenu></SideMenu> */}
        <PageContent></PageContent>
      </div>
      <AppFooter />
    </div>
  );
}
export default App;