import { NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import { join, dirname, basename, isAbsolute, resolve, sep } from 'path';
import { existsSync } from 'fs';

const DEFAULT_WORKSPACE_PATH = process.env.WORKSPACE_PATH || './documents';

interface DirectoryEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
}

// The vault root is the only directory tree the workspace API may expose.
function getVaultRoot(): string {
  return resolve(process.cwd(), DEFAULT_WORKSPACE_PATH);
}

function isWithin(root: string, target: string): boolean {
  return target === root || target.startsWith(root + sep);
}

// Resolve a requested path and confirm it stays inside the vault.
// Returns null when the path is invalid or escapes the vault.
function resolveVaultPath(requested: string): string | null {
  if (requested.split(/[\\/]+/).some((segment) => segment === '..')) {
    return null;
  }
  const resolved = isAbsolute(requested)
    ? resolve(requested)
    : resolve(process.cwd(), requested);
  if (!isWithin(getVaultRoot(), resolved)) {
    return null;
  }
  return resolved;
}

// Get list of directories for folder browser (confined to the vault root)
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const vaultRoot = getVaultRoot();
  const requested = searchParams.get('path');
  const path = requested ? resolveVaultPath(requested) : vaultRoot;

  if (!path) {
    return NextResponse.json(
      { error: 'Path is outside the workspace vault' },
      { status: 403 }
    );
  }

  try {
    // Validate path exists
    if (!existsSync(path)) {
      return NextResponse.json(
        { error: 'Path does not exist', path },
        { status: 404 }
      );
    }

    const stats = await stat(path);
    if (!stats.isDirectory()) {
      return NextResponse.json(
        { error: 'Path is not a directory', path },
        { status: 400 }
      );
    }

    const items = await readdir(path);
    const entries: DirectoryEntry[] = [];

    for (const item of items) {
      // Skip hidden files/folders on Windows and Unix
      if (item.startsWith('.') || item.startsWith('$')) continue;

      try {
        const fullPath = join(path, item);
        const itemStats = await stat(fullPath);

        // Only include directories
        if (itemStats.isDirectory()) {
          entries.push({
            name: item,
            path: fullPath,
            type: 'directory',
          });
        }
      } catch {
        // Skip items we can't access
        continue;
      }
    }

    // Sort alphabetically
    entries.sort((a, b) => a.name.localeCompare(b.name));

    const canGoUp = path !== vaultRoot; // Never browse above the vault root

    return NextResponse.json({
      currentPath: path,
      parentPath: canGoUp ? dirname(path) : path,
      entries,
      canGoUp,
    });
  } catch (error) {
    console.error('Error browsing directory:', error);
    return NextResponse.json(
      { error: 'Failed to browse directory' },
      { status: 500 }
    );
  }
}

// Validate a workspace path (must stay inside the vault)
export async function POST(request: Request) {
  try {
    const { path } = await request.json();

    if (!path) {
      return NextResponse.json(
        { error: 'Path is required' },
        { status: 400 }
      );
    }

    const resolvedPath = resolveVaultPath(path);
    if (!resolvedPath) {
      return NextResponse.json(
        { valid: false, error: 'Path is outside the workspace vault', path },
        { status: 403 }
      );
    }

    // Check if path exists
    if (!existsSync(resolvedPath)) {
      return NextResponse.json({
        valid: false,
        error: 'Path does not exist',
        path,
      });
    }

    // Check if it's a directory
    const stats = await stat(resolvedPath);
    if (!stats.isDirectory()) {
      return NextResponse.json({
        valid: false,
        error: 'Path is not a directory',
        path,
      });
    }

    // Count markdown files
    let markdownCount = 0;
    const countMarkdown = async (dir: string, depth = 0) => {
      if (depth > 3) return; // Don't go too deep

      try {
        const items = await readdir(dir);
        for (const item of items) {
          if (item.startsWith('.') || item === 'node_modules') continue;

          const fullPath = join(dir, item);
          try {
            const itemStats = await stat(fullPath);
            if (itemStats.isDirectory()) {
              await countMarkdown(fullPath, depth + 1);
            } else if (item.match(/\.(md|markdown|mdx|txt)$/i)) {
              markdownCount++;
            }
          } catch {
            continue;
          }
        }
      } catch {
        // Ignore errors
      }
    };

    await countMarkdown(resolvedPath);

    return NextResponse.json({
      valid: true,
      path: resolvedPath,
      name: basename(resolvedPath),
      markdownFiles: markdownCount,
    });
  } catch (error) {
    console.error('Error validating workspace:', error);
    return NextResponse.json(
      { error: 'Failed to validate workspace' },
      { status: 500 }
    );
  }
}
