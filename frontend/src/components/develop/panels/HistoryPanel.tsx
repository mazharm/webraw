import { PanelSection } from '../../common/PanelSection';
import { tokens, Text, Button, Input } from '@fluentui/react-components';
import { useEditStore } from '../../../stores/editStore';
import { useState, useMemo } from 'react';

export function HistoryPanel() {
  const editState = useEditStore(s => s.editState);
  const setEditState = useEditStore(s => s.setEditState);
  const createSnapshot = useEditStore(s => s.createSnapshot);
  const [snapshotName, setSnapshotName] = useState('');

  if (!editState) return null;

  const historyList = useMemo(() => {
    const nodes = Object.values(editState.history.nodes);
    return nodes.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());
  }, [editState.history.nodes]);

  const handleRestoreHistory = (nodeId: string) => {
    const node = editState.history.nodes[nodeId];
    if (!node) return;
    setEditState({
      ...node.snapshot,
      history: { ...editState.history, headId: nodeId },
      snapshots: editState.snapshots,
    } as any);
  };

  const handleCreateSnapshot = () => {
    if (snapshotName.trim()) {
      createSnapshot(snapshotName.trim());
      setSnapshotName('');
    }
  };

  return (
    <PanelSection title="History / Snapshots">
      {/* Snapshots */}
      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>Snapshots</Text>
      <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
        <Input
          size="small"
          placeholder="Snapshot name"
          value={snapshotName}
          onChange={(_, data) => setSnapshotName(data.value)}
          style={{ flex: 1 }}
        />
        <Button size="small" onClick={handleCreateSnapshot} disabled={!snapshotName.trim()}>
          Save
        </Button>
      </div>

      {editState.snapshots.map(snap => (
        <div
          key={snap.id}
          style={{
            padding: '4px 8px',
            fontSize: 11,
            cursor: 'pointer',
            background: tokens.colorNeutralBackground3,
            borderRadius: 4,
            marginBottom: 2,
          }}
          onClick={() => {
            const targetNode = Object.values(editState.history.nodes)
              .find(n => n.id === snap.stateHash);
            if (targetNode) handleRestoreHistory(targetNode.id);
          }}
        >
          {snap.name}
          <span style={{ color: '#666', marginLeft: 8, fontSize: 10 }}>
            {new Date(snap.createdAt).toLocaleTimeString()}
          </span>
        </div>
      ))}

      <div style={{ height: 8 }} />

      {/* History */}
      <Text size={200} weight="semibold" style={{ display: 'block', marginBottom: 4 }}>
        History ({historyList.length})
      </Text>
      <div style={{ maxHeight: 200, overflowY: 'auto' }}>
        {historyList.map(node => (
          <div
            key={node.id}
            onClick={() => handleRestoreHistory(node.id)}
            style={{
              padding: '3px 8px',
              fontSize: 11,
              cursor: 'pointer',
              background: node.id === editState.history.headId
                ? tokens.colorBrandBackground
                : 'transparent',
              borderRadius: 3,
              color: node.id === editState.history.headId
                ? tokens.colorNeutralForegroundOnBrand
                : tokens.colorNeutralForeground2,
            }}
          >
            {node.label}
          </div>
        ))}
      </div>
    </PanelSection>
  );
}
