import { NextResponse } from 'next/server';
import { readFile, writeFile, unlink, rename } from 'fs/promises';
import { join, dirname, basename, isAbsolute, resolve, sep } from 'path';
import { existsSync } from 'fs';

const DEFAULT_WORKSPACE_PATH = process.env.WORKSPACE_PATH || './documents';

// Thrown when a requested path would escape the configured vault root.
class PathValidationError extends Error {
  status: number;
  constructor(message: string, status = 403) {
    super(message);
    this.status = status;
  }
}

// The vault root is the only directory tree the file API may touch.
function getVaultRoot(): string {
  return resolve(process.cwd(), DEFAULT_WORKSPACE_PATH);
}

function isWithin(root: string, target: string): boolean {
  return target === root || target.startsWith(root + sep);
}

// Resolve the workspace root, rejecting values that escape the vault.
function resolveWorkspaceRoot(customPath?: string): string {
  const vaultRoot = getVaultRoot();
  if (!customPath) {
    return vaultRoot;
  }
  const resolvedWorkspace = isAbsolute(customPath)
    ? resolve(customPath)
    : resolve(process.cwd(), customPath);
  if (!isWithin(vaultRoot, resolvedWorkspace)) {
    throw new PathValidationError('Workspace path escapes the vault');
  }
  return resolvedWorkspace;
}

function getFullPath(pathSegments: string[], workspace?: string): string {
  const resolvedRoot = resolveWorkspaceRoot(workspace);
  const relativePath = pathSegments.join('/');

  // Reject any ".." path segment outright.
  if (relativePath.split(/[\\/]+/).some((segment) => segment === '..')) {
    throw new PathValidationError('Path traversal segments are not allowed', 400);
  }

  const fullPath = resolve(resolvedRoot, relativePath);
  if (!fullPath.startsWith(resolvedRoot + sep)) {
    throw new PathValidationError('Resolved path escapes the workspace');
  }
  return fullPath;
}

function pathErrorResponse(error: unknown): NextResponse | null {
  if (error instanceof PathValidationError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  return null;
}

export async function GET(
  request: Request,
  { params }: { params: { path: string[] } }
) {
  try {
    const { searchParams } = new URL(request.url);
    const workspace = searchParams.get('workspace') || undefined;
    const filePath = getFullPath(params.path, workspace);

    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    const content = await readFile(filePath, 'utf-8');
    const path = params.path.join('/');

    return NextResponse.json({
      content,
      path,
      name: basename(filePath),
    });
  } catch (error) {
    const rejected = pathErrorResponse(error);
    if (rejected) return rejected;
    console.error('Error reading file:', error);
    return NextResponse.json(
      { error: 'Failed to read file' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: Request,
  { params }: { params: { path: string[] } }
) {
  try {
    const { content, workspace } = await request.json();
    const filePath = getFullPath(params.path, workspace);

    // Ensure the directory exists
    const { mkdir } = await import('fs/promises');
    await mkdir(dirname(filePath), { recursive: true });

    await writeFile(filePath, content, 'utf-8');

    return NextResponse.json({
      success: true,
      path: params.path.join('/'),
    });
  } catch (error) {
    const rejected = pathErrorResponse(error);
    if (rejected) return rejected;
    console.error('Error saving file:', error);
    return NextResponse.json(
      { error: 'Failed to save file' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: { path: string[] } }
) {
  try {
    const { searchParams } = new URL(request.url);
    const workspace = searchParams.get('workspace') || undefined;
    const filePath = getFullPath(params.path, workspace);

    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    await unlink(filePath);

    return NextResponse.json({
      success: true,
      path: params.path.join('/'),
    });
  } catch (error) {
    const rejected = pathErrorResponse(error);
    if (rejected) return rejected;
    console.error('Error deleting file:', error);
    return NextResponse.json(
      { error: 'Failed to delete file' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: Request,
  { params }: { params: { path: string[] } }
) {
  try {
    const { newName, workspace } = await request.json();
    const oldPath = getFullPath(params.path, workspace);

    // The new name must be a plain file name, not a path.
    if (
      typeof newName !== 'string' ||
      newName.length === 0 ||
      newName.includes('/') ||
      newName.includes('\\') ||
      newName === '..' ||
      newName === '.'
    ) {
      return NextResponse.json(
        { error: 'Invalid file name' },
        { status: 400 }
      );
    }

    if (!existsSync(oldPath)) {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    const newPath = join(dirname(oldPath), newName);
    const resolvedRoot = resolveWorkspaceRoot(workspace);
    if (!resolve(newPath).startsWith(resolvedRoot + sep)) {
      return NextResponse.json(
        { error: 'Resolved path escapes the workspace' },
        { status: 403 }
      );
    }

    if (existsSync(newPath)) {
      return NextResponse.json(
        { error: 'A file with that name already exists' },
        { status: 409 }
      );
    }

    await rename(oldPath, newPath);

    const pathParts = [...params.path];
    pathParts[pathParts.length - 1] = newName;

    return NextResponse.json({
      success: true,
      oldPath: params.path.join('/'),
      newPath: pathParts.join('/'),
    });
  } catch (error) {
    const rejected = pathErrorResponse(error);
    if (rejected) return rejected;
    console.error('Error renaming file:', error);
    return NextResponse.json(
      { error: 'Failed to rename file' },
      { status: 500 }
    );
  }
}
