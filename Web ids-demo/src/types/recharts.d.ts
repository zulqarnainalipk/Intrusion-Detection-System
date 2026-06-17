// Type declarations for Recharts to fix React 18.3 compatibility
declare module 'recharts' {
  import { ComponentType } from 'react';

  export interface ResponsiveContainerProps {
    width: string | number;
    height: string | number;
    children: React.ReactNode;
  }

  export const ResponsiveContainer: ComponentType<ResponsiveContainerProps>;

  export interface PieProps {
    data?: any[];
    cx?: string | number;
    cy?: string | number;
    innerRadius?: string | number;
    outerRadius?: string | number;
    paddingAngle?: number;
    dataKey: string;
    animationDuration?: number;
    animationBegin?: number;
    children?: React.ReactNode;
  }

  export const Pie: ComponentType<PieProps>;

  export interface CellProps {
    fill?: string;
    stroke?: string;
  }

  export const Cell: ComponentType<CellProps>;

  export interface TooltipProps {
    contentStyle?: React.CSSProperties;
    formatter?: (value: any, name: any) => any;
    labelStyle?: React.CSSProperties;
  }

  export const Tooltip: ComponentType<TooltipProps>;

  export interface XAxisProps {
    dataKey?: string;
    type?: 'number' | 'category';
    stroke?: string;
    tick?: any;
    domain?: any[];
    width?: number;
  }

  export const XAxis: ComponentType<XAxisProps>;

  export interface YAxisProps {
    dataKey?: string;
    type?: 'number' | 'category';
    stroke?: string;
    tick?: any;
    domain?: any[];
    width?: number;
  }

  export const YAxis: ComponentType<YAxisProps>;

  export interface CartesianGridProps {
    strokeDasharray?: string;
    stroke?: string;
    horizontal?: boolean;
  }

  export const CartesianGrid: ComponentType<CartesianGridProps>;

  export interface LegendProps {}

  export const Legend: ComponentType<LegendProps>;

  export interface AreaProps {
    type?: string;
    dataKey: string;
    stroke?: string;
    fillOpacity?: number;
    fill?: string;
    name?: string;
    strokeWidth?: number;
  }

  export const Area: ComponentType<AreaProps>;

  export interface BarProps {
    dataKey: string;
    fill?: string;
    radius?: number[];
    animationDuration?: number;
    children?: React.ReactNode;
  }

  export const Bar: ComponentType<BarProps>;

  export interface LineProps {
    type?: string;
    dataKey: string;
    stroke?: string;
    strokeWidth?: number;
    name?: string;
    animationDuration?: number;
  }

  export const Line: ComponentType<LineProps>;

  export interface PieChartProps {
    children?: React.ReactNode;
  }

  export const PieChart: ComponentType<PieChartProps>;

  export interface AreaChartProps {
    data?: any[];
    margin?: any;
    children?: React.ReactNode;
  }

  export const AreaChart: ComponentType<AreaChartProps>;

  export interface BarChartProps {
    data?: any[];
    layout?: 'vertical' | 'horizontal';
    margin?: any;
    children?: React.ReactNode;
  }

  export const BarChart: ComponentType<BarChartProps>;

  export interface LineChartProps {
    data?: any[];
    margin?: any;
    children?: React.ReactNode;
  }

  export const LineChart: ComponentType<LineChartProps>;
}
