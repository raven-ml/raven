import * as esbuild from 'esbuild';

const production = process.argv.includes('--production');

const config = {
  entryPoints: ['src/app.js'],
  bundle: true,
  minify: production,
  sourcemap: !production,
  outdir: 'dist',
  format: 'esm',
  target: 'es2020',
  loader: { '.css': 'css', '.woff2': 'file' },
  external: ['/assets/*'],
};

if (process.argv.includes('--watch')) {
  const ctx = await esbuild.context(config);
  await ctx.watch();
  console.log('Watching for changes...');
} else {
  await esbuild.build(config);
}
