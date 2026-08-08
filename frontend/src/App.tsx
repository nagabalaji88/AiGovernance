import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { motion } from 'framer-motion';

import Engineering from './pages/Engineering';
import Executive from './pages/Executive';
import Governance from './pages/Governance';
import Optimization from './pages/Optimization';
import Simulator from './pages/Simulator';

const NAV = [
  { to: '/executive', label: 'Executive', hint: 'Spend, trajectory, opportunity' },
  { to: '/engineering', label: 'Engineering', hint: 'Token and prompt analytics' },
  { to: '/optimization', label: 'Optimization', hint: 'Ranked recommendations' },
  { to: '/governance', label: 'Governance', hint: 'Budgets, chargeback, anomalies' },
  { to: '/simulator', label: 'Simulator', hint: 'What-if analysis' },
];

export default function App() {
  return (
    <div className="min-h-screen">
      <div className="aurora" aria-hidden="true" />

      {/* Skip link: the first tab stop on every page. Without it, keyboard
          users traverse the entire nav on every navigation. */}
      <a
        href="#main"
        className="focus-ring sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-slate-800 focus:px-4 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <div className="mx-auto flex max-w-[100rem] gap-6 px-4 py-6 lg:px-8">
        <Sidebar />
        <main id="main" className="min-w-0 flex-1 pb-16">
          <Routes>
            <Route path="/" element={<Navigate to="/executive" replace />} />
            <Route path="/executive" element={<Executive />} />
            <Route path="/engineering" element={<Engineering />} />
            <Route path="/optimization" element={<Optimization />} />
            <Route path="/governance" element={<Governance />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route
              path="*"
              element={
                <div className="glass p-8 text-center">
                  <p className="text-sm text-slate-300">That page does not exist.</p>
                </div>
              }
            />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="sticky top-6 hidden h-fit w-60 shrink-0 lg:block">
      <div className="glass p-4">
        <div className="flex items-center gap-2.5 px-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white">
            AI
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-100">Cost Intelligence</p>
            <p className="truncate text-[11px] text-slate-500">Token optimization platform</p>
          </div>
        </div>

        <nav className="mt-5 space-y-1" aria-label="Dashboards">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                [
                  'focus-ring relative block rounded-lg px-3 py-2 transition-colors',
                  isActive ? 'text-slate-50' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200',
                ].join(' ')
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-lg bg-indigo-500/15 ring-1 ring-indigo-400/25"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className="relative block text-sm font-medium">{item.label}</span>
                  <span className="relative block text-[11px] text-slate-500">{item.hint}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <p className="mt-4 px-2 text-[11px] leading-relaxed text-slate-600">
        Costs resolve against effective-dated provider rate cards. Historical figures never restate
        when prices change.
      </p>
    </aside>
  );
}
