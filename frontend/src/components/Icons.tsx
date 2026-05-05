import React from 'react';

type IconProps = { size?: number; stroke?: number; className?: string };
const base = (size: number, stroke: number) => ({
  width: size, height: size,
  viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: stroke,
  strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const,
});

export const IChat = ({ size = 16, stroke = 1.7, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
);
export const IDoc = ({ size = 16, stroke = 1.7, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="13" y2="17"/></svg>
);
export const ITable = ({ size = 16, stroke = 1.7, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><rect x="3" y="3" width="18" height="18" rx="1.5"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
);
export const ISend = ({ size = 16, stroke = 1.8, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
);
export const IUpload = ({ size = 16, stroke = 1.7, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
);
export const ITrash = ({ size = 14, stroke = 1.7, className }: IconProps) => (
  <svg {...base(size, stroke)} className={className}><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
);
