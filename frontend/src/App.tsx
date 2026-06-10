import { Route, Routes, Link } from "react-router-dom";

import { BriefPage } from "@/pages/Brief";
import { Home } from "@/pages/Home";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-baseline gap-3 px-4 py-3">
          <Link to="/" className="font-display text-xl font-bold tracking-tight text-slate-900">
            Briefr
          </Link>
          <p className="text-xs text-slate-500">
            Sourced research briefs with per-section faithfulness scores
          </p>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/briefs/:id" element={<BriefPage />} />
      </Routes>
    </div>
  );
}
