import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

import App from './App';
import './styles/index.css';

/**
 * Query defaults tuned for a cost dashboard.
 *
 * `staleTime` of 60s: usage data is aggregated on a rolling basis and does not
 * change meaningfully second to second. Refetching on every window focus — the
 * library default — means an analyst alt-tabbing between this and a
 * spreadsheet triggers a burst of aggregation queries, which is both wasteful
 * and, ironically, expensive.
 *
 * Retries are capped at 1. A failing aggregation query is usually failing for
 * a reason that a retry will not fix (bad window, missing permission), and
 * three silent retries just delay the error message the user needs to see.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
