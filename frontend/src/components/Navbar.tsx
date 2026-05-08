import { NavLink } from "react-router-dom";

function Navbar() {
  return (
    <header className="navbar">
      <NavLink className="brand" to="/">
        <span className="brand-mark">DL</span>
        <span>DeepLens AI</span>
      </NavLink>
      <nav className="nav-links">
        <NavLink to="/">首页</NavLink>
        <NavLink to="/upload">上传检测</NavLink>
        <NavLink to="/result">结果页</NavLink>
        <NavLink to="/explain">可解释化</NavLink>
      </nav>
    </header>
  );
}

export default Navbar;
