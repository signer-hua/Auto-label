/**
 * 全局类别管理面板
 * 所有模式共享的类别定义（名称+颜色），持久化到 localStorage。
 * 固定显示在工具栏底部，支持添加/编辑/删除类别。
 */
import React, { useRef, useState } from 'react';
import { Button, Tag, Popconfirm, Input, Tooltip } from 'antd';
import { PlusOutlined, DeleteOutlined, EditOutlined, CheckOutlined } from '@ant-design/icons';
import { useAppStore } from '../store/useAppStore';
import { getNextAvailableColor } from '../utils/categoryColorMap';

const CategoryPanel: React.FC = () => {
  const {
    categories, activeCategoryId,
    addCategory, updateCategory, removeCategory, setActiveCategoryId,
  } = useAppStore();

  const nameRef = useRef<HTMLInputElement>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editColor, setEditColor] = useState('#FF5050');

  const handleAdd = () => {
    const name = nameRef.current?.value?.trim();
    if (!name) return;
    const usedColors = categories.map(c => c.color);
    const color = getNextAvailableColor(usedColors);
    addCategory(name, color);
    if (nameRef.current) nameRef.current.value = '';
  };

  const startEdit = (id: string, name: string, color: string) => {
    setEditingId(id);
    setEditName(name);
    setEditColor(color);
  };

  const confirmEdit = () => {
    if (editingId && editName.trim()) {
      updateCategory(editingId, editName.trim(), editColor);
    }
    setEditingId(null);
  };

  return (
    <div style={{ borderTop: '1px solid #444', padding: '8px 0' }}>
      <div style={{ color: '#999', fontSize: 12, marginBottom: 6 }}>
        全局类别管理（所有模式共用）
      </div>

      {/* 添加类别 */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
        <input
          ref={nameRef}
          type="text"
          placeholder="输入类别名（如 cat）"
          style={{
            flex: 1, background: '#2a2a2a', border: '1px solid #444',
            color: '#ddd', padding: '3px 6px', borderRadius: 4, fontSize: 12,
          }}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
        />
        <Button size="small" icon={<PlusOutlined />} onClick={handleAdd} />
      </div>

      {/* 类别列表 */}
      {categories.map((cat) => {
        const isActive = cat.id === activeCategoryId;
        const isEditing = editingId === cat.id;

        if (isEditing) {
          return (
            <div key={cat.id} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3, padding: '2px 4px' }}>
              <input
                type="color"
                value={editColor}
                onChange={(e) => setEditColor(e.target.value)}
                style={{ width: 22, height: 22, padding: 0, border: 'none', cursor: 'pointer' }}
              />
              <Input
                size="small"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onPressEnter={confirmEdit}
                style={{ flex: 1, background: '#2a2a2a', borderColor: '#555', color: '#ddd' }}
              />
              <Button size="small" type="primary" icon={<CheckOutlined />} onClick={confirmEdit}
                style={{ width: 24, height: 24, padding: 0, minWidth: 24 }} />
            </div>
          );
        }

        return (
          <div
            key={cat.id}
            onClick={() => setActiveCategoryId(cat.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5, marginBottom: 3,
              padding: '3px 6px', borderRadius: 4, cursor: 'pointer',
              background: isActive ? '#333' : 'transparent',
              border: isActive ? `1px solid ${cat.color}` : '1px solid transparent',
            }}
          >
            <div style={{ width: 12, height: 12, borderRadius: 2, background: cat.color, flexShrink: 0 }} />
            <span style={{ color: '#ddd', fontSize: 12, flex: 1 }}>{cat.name}</span>
            <Tooltip title="编辑">
              <Button type="text" size="small" icon={<EditOutlined style={{ fontSize: 10, color: '#888' }} />}
                onClick={(e) => { e.stopPropagation(); startEdit(cat.id, cat.name, cat.color); }}
                style={{ width: 20, height: 20, padding: 0, minWidth: 20 }} />
            </Tooltip>
            <Popconfirm title={`删除类别「${cat.name}」？`} onConfirm={() => removeCategory(cat.id)} okText="删除" cancelText="取消">
              <Button type="text" size="small" danger icon={<DeleteOutlined style={{ fontSize: 10 }} />}
                onClick={(e) => e.stopPropagation()}
                style={{ width: 20, height: 20, padding: 0, minWidth: 20 }} />
            </Popconfirm>
          </div>
        );
      })}

      {categories.length === 0 && (
        <div style={{ color: '#555', fontSize: 11, textAlign: 'center', padding: '4px 0' }}>
          暂无类别，请添加
        </div>
      )}
    </div>
  );
};

export default CategoryPanel;
