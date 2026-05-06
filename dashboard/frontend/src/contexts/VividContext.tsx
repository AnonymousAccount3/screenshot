import { createContext, useContext } from 'react';

export const VividContext = createContext(false);

export function useVivid(): boolean {
  return useContext(VividContext);
}
