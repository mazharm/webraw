import { tokens, Text } from '@fluentui/react-components';
import { ChevronDownRegular, ChevronRightRegular } from '@fluentui/react-icons';
import { useState, type ReactNode } from 'react';

interface Props {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function PanelSection({ title, defaultOpen = false, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div style={{ borderBottom: `1px solid ${tokens.colorNeutralStroke2}` }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          width: '100%',
          padding: '8px 12px',
          border: 'none',
          background: 'transparent',
          cursor: 'pointer',
          color: tokens.colorNeutralForeground1,
          fontSize: 12,
          fontWeight: 600,
        }}
        aria-expanded={open}
      >
        {open ? <ChevronDownRegular fontSize={12} /> : <ChevronRightRegular fontSize={12} />}
        {title}
      </button>
      {open && (
        <div style={{ padding: '0 12px 8px' }}>
          {children}
        </div>
      )}
    </div>
  );
}
