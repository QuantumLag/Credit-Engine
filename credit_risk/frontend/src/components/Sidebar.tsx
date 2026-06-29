import { Link, useRouterState } from "@tanstack/react-router";

function Icon({ d }: { d: string }) {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

export function Sidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-mark">⊙</span>
        CreditLens
      </div>
      <Link to="/" className={`nav-item ${pathname === "/" ? "active" : ""}`}>
        <Icon d="M3 12l2-2 4 4 8-8 4 4M3 21h18" />
        Portfolio Overview
      </Link>
      <Link to="/applicants" className={`nav-item ${pathname.startsWith("/applicants") ? "active" : ""}`}>
        <Icon d="M21 21l-4.35-4.35M10 17a7 7 0 100-14 7 7 0 000 14z" />
        Applicant Lookup
      </Link>
    </aside>
  );
}
