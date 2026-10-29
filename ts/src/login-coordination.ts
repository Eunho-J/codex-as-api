// One active login per resolved store within the supported single-server process.
const active = new Map<string, object>();
export function reserveLogin(path: string, owner: object): boolean {
  if (active.has(path)) return false;
  active.set(path, owner);
  return true;
}
export function releaseLogin(path: string, owner: object): void {
  if (active.get(path) === owner) active.delete(path);
}
