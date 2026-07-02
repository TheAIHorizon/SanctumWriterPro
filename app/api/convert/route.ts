import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { join, isAbsolute, resolve, sep } from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

const DEFAULT_WORKSPACE_PATH = process.env.WORKSPACE_PATH || './documents';

// The vault root is the only directory tree conversion output may be written to.
function getVaultRoot(): string {
  return resolve(process.cwd(), DEFAULT_WORKSPACE_PATH);
}

// Resolve a caller-supplied output directory, returning null if it escapes the vault.
function resolveOutputDir(requested: string): string | null {
  if (requested.split(/[\\/]+/).some((segment) => segment === '..')) {
    return null;
  }
  const vaultRoot = getVaultRoot();
  const resolved = isAbsolute(requested)
    ? resolve(requested)
    : resolve(process.cwd(), requested);
  if (resolved !== vaultRoot && !resolved.startsWith(vaultRoot + sep)) {
    return null;
  }
  return resolved;
}

interface ConversionResult {
  success: boolean;
  markdown?: string;
  metadata?: {
    title: string;
    pages?: number;
    source_file?: string;
  };
  saved_path?: string;
  character_count?: number;
  word_count?: number;
  error?: string;
}

async function runPythonScript(inputPath: string, outputPath?: string): Promise<ConversionResult> {
  return new Promise((resolve) => {
    const args = ['scripts/convert_document.py', inputPath];
    if (outputPath) {
      args.push(outputPath);
    }
    
    const python = spawn('python', args, {
      cwd: process.cwd(),
    });
    
    let stdout = '';
    let stderr = '';
    
    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    python.on('close', (code) => {
      if (code !== 0) {
        // Try to parse error from stdout first (our script outputs JSON errors)
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch {
          resolve({
            success: false,
            error: stderr || `Python script exited with code ${code}`,
          });
        }
        return;
      }
      
      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        resolve({
          success: false,
          error: 'Failed to parse conversion result',
        });
      }
    });
    
    python.on('error', (err) => {
      resolve({
        success: false,
        error: `Failed to run Python: ${err.message}. Make sure Python is installed.`,
      });
    });
  });
}

export async function POST(request: Request) {
  let tempInputPath: string | null = null;
  
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const outputPath = formData.get('outputPath') as string | null;
    const filename = formData.get('filename') as string | null;
    
    if (!file) {
      return NextResponse.json(
        { success: false, error: 'No file provided' },
        { status: 400 }
      );
    }
    
    // Save uploaded file to temp location
    const tempDir = join(tmpdir(), 'sanctum-convert');
    await mkdir(tempDir, { recursive: true });
    
    const tempFilename = `${randomUUID()}-${file.name}`;
    tempInputPath = join(tempDir, tempFilename);
    
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(tempInputPath, buffer);
    
    // Determine output path (confined to the workspace vault)
    let fullOutputPath: string | undefined;
    if (outputPath) {
      const outputDir = resolveOutputDir(outputPath);
      if (!outputDir) {
        return NextResponse.json(
          { success: false, error: 'Output path is outside the workspace vault' },
          { status: 403 }
        );
      }

      const outputFilename = filename || `${file.name.replace(/\.[^/.]+$/, '')}.md`;
      // The output file name must be a plain name, not a path.
      if (
        outputFilename.includes('/') ||
        outputFilename.includes('\\') ||
        outputFilename === '..' ||
        outputFilename === '.'
      ) {
        return NextResponse.json(
          { success: false, error: 'Invalid output file name' },
          { status: 400 }
        );
      }

      fullOutputPath = join(outputDir, outputFilename);
    }
    
    // Run conversion
    const result = await runPythonScript(tempInputPath, fullOutputPath);
    
    // Cleanup temp file
    try {
      await unlink(tempInputPath);
    } catch {
      // Ignore cleanup errors
    }
    
    if (!result.success) {
      return NextResponse.json(result, { status: 500 });
    }
    
    return NextResponse.json(result);
    
  } catch (error) {
    // Cleanup temp file on error
    if (tempInputPath) {
      try {
        await unlink(tempInputPath);
      } catch {
        // Ignore cleanup errors
      }
    }
    
    console.error('Conversion error:', error);
    return NextResponse.json(
      { success: false, error: 'Conversion failed' },
      { status: 500 }
    );
  }
}

// Health check - verify Python and Docling are available
export async function GET() {
  return new Promise<Response>((resolve) => {
    const python = spawn('python', ['-c', 'import docling; print("ok")'], {
      cwd: process.cwd(),
    });
    
    let stdout = '';
    
    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0 && stdout.trim() === 'ok') {
        resolve(NextResponse.json({ available: true }));
      } else {
        resolve(NextResponse.json({ 
          available: false,
          hint: 'Run: pip install -r requirements.txt'
        }));
      }
    });
    
    python.on('error', () => {
      resolve(NextResponse.json({ 
        available: false,
        hint: 'Python not found. Make sure Python is installed and in PATH.'
      }));
    });
  });
}
