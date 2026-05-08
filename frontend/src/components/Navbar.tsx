import { NavLink } from "react-router-dom";

function Navbar() {
  return (
    <header className="navbar">
      <NavLink className="brand" to="/">
        <span className="brand-mark">DL</span>
        <span>DeepLens AI</span>
      </NavLink>

      <nav className="nav-links">
        <NavLink to="/">Home</NavLink>
        <NavLink to="/upload">Upload</NavLink>
        <NavLink to="/result">Result</NavLink>
        <NavLink to="/explain">Explain</NavLink>
      </nav>
    </header>
  );
}

export default Navbar;
