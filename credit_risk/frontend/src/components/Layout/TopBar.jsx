export default function TopBar({ title }) {
  return (
    <header className="topbar">
      <h1 className="topbar-title">{title}</h1>
      <span className="demo-badge">Demo Mode — 10 sample applicants</span>
    </header>
  );
}
