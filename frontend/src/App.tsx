import { FluentProvider, webLightTheme, webDarkTheme } from '@fluentui/react-components';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from './components/common/AppShell';
import { useState } from 'react';
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

  return (
    <FluentProvider theme={darkMode ? webDarkTheme : webLightTheme} style={{ height: '100%' }}>
      <QueryClientProvider client={queryClient}>
        <AppShell />
      </QueryClientProvider>
    </FluentProvider>
  );
}

export default App;
