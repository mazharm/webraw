import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from './components/common/AppShell';
import { useEffect, useState } from 'react';
import { healthCheck } from './api/client';
import { useSettingsStore } from './stores/settingsStore';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000,
      retry: 2,
    },
  },
});

function App() {
  const [darkMode] = useState(true);
  const setBackendHealthy = useSettingsStore(s => s.setBackendHealthy);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setBackendHealthy(true);
      } catch {
        setBackendHealthy(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [setBackendHealthy]);

  return (
    <FluentProvider theme={darkMode ? webDarkTheme : webLightTheme}>
      <QueryClientProvider client={queryClient}>
        <AppShell />
      </QueryClientProvider>
    </FluentProvider>
  );
}

export default App;
