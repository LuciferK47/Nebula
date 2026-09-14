import React, { createContext, useContext, useEffect, useState } from 'react';
import { api, type SystemInfo } from './api';

export type LiveModeState =
  | { status: 'probing' }
  | { status: 'live'; info: SystemInfo }
  | { status: 'static' };

const LiveModeContext = createContext<LiveModeState>({ status: 'probing' });

export function LiveModeProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<LiveModeState>({ status: 'probing' });

  useEffect(() => {
    let active = true;

    async function checkServer() {
      try {
        const info = await api.systemInfo();
        if (active) {
          setState({ status: 'live', info });
        }
      } catch {
        if (active) {
          setState({ status: 'static' });
        }
      }
    }

    checkServer();

    // Re-check periodically if static or probing
    const timer = setInterval(() => {
      checkServer();
    }, 10000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <LiveModeContext.Provider value={state}>
      {children}
    </LiveModeContext.Provider>
  );
}

export function useLiveMode(): LiveModeState {
  return useContext(LiveModeContext);
}
