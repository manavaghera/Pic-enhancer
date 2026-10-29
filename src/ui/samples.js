// Procedurally drawn sample images, deliberately degraded (small, soft, noisy,
// heavily compressed) so the enhancer has something real to fix. No bundled
// photos, so there are no licensing questions.

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

const toFile = (c, name, type, quality) =>
  new Promise((resolve, reject) => {
    c.toBlob((blob) => (blob ? resolve(new File([blob], name, { type })) : reject(new Error('Could not create sample'))), type, quality);
  });

function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function addGrain(ctx, w, h, amount, seed) {
  const rand = seeded(seed);
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let p = 0; p < d.length; p += 4) {
    const n = (rand() + rand() + rand() - 1.5) * amount;
    d[p] += n + (rand() - 0.5) * amount * 0.6;
    d[p + 1] += n;
    d[p + 2] += n + (rand() - 0.5) * amount * 0.6;
  }
  ctx.putImageData(img, 0, 0);
}

function landscape() {
  const W = 960, H = 640;
  const big = canvas(W, H);
  const g = big.getContext('2d');
  const sky = g.createLinearGradient(0, 0, 0, H * 0.6);
  sky.addColorStop(0, '#2b4a8a');
  sky.addColorStop(0.6, '#e98a5a');
  sky.addColorStop(1, '#f6c27a');
  g.fillStyle = sky;
  g.fillRect(0, 0, W, H);
  g.fillStyle = '#fff3c4';
  g.beginPath();
  g.arc(W * 0.68, H * 0.42, 46, 0, Math.PI * 2);
  g.fill();
  const ridge = (color, base, amp, freq, phase) => {
    g.fillStyle = color;
    g.beginPath();
    g.moveTo(0, H);
    for (let x = 0; x <= W; x += 8) g.lineTo(x, base + Math.sin(x * freq + phase) * amp + Math.sin(x * freq * 2.7) * amp * 0.35);
    g.lineTo(W, H);
    g.fill();
  };
  ridge('#6b4e7a', H * 0.5, 40, 0.008, 1);
  ridge('#3e3459', H * 0.58, 30, 0.013, 2);
  const lake = g.createLinearGradient(0, H * 0.7, 0, H);
  lake.addColorStop(0, '#d9875f');
  lake.addColorStop(1, '#2a3560');
  g.fillStyle = lake;
  g.fillRect(0, H * 0.7, W, H * 0.3);
  const rand = seeded(42);
  g.fillStyle = '#1d2a22';
  for (let i = 0; i < 26; i++) {
    const x = rand() * W, base = H * 0.72, th = 50 + rand() * 90;
    g.beginPath();
    g.moveTo(x, base - th);
    g.lineTo(x - th * 0.22, base);
    g.lineTo(x + th * 0.22, base);
    g.fill();
  }
  g.strokeStyle = 'rgba(255,240,210,0.6)';
  g.lineWidth = 2;
  for (let i = 0; i < 14; i++) {
    const y = H * 0.76 + i * 11;
    g.beginPath();
    g.moveTo(W * 0.6 - 60 + i * 3, y);
    g.lineTo(W * 0.76 - i * 3, y);
    g.stroke();
  }
  const small = canvas(320, 213);
  const s = small.getContext('2d');
  s.imageSmoothingQuality = 'low';
  s.drawImage(big, 0, 0, 320, 213);
  addGrain(s, 320, 213, 26, 7);
  return toFile(small, 'sample-sunset.jpg', 'image/jpeg', 0.42);
}

function documentScan() {
  const W = 520, H = 360;
  const c = canvas(W, H);
  const g = c.getContext('2d');
  g.fillStyle = '#efe9dc';
  g.fillRect(0, 0, W, H);
  g.fillStyle = '#2a2622';
  g.font = 'bold 22px Georgia, serif';
  g.fillText('Field Notes — Day 14', 34, 52);
  g.font = '15px Georgia, serif';
  const lines = [
    'Morning fog lifted by nine. The ridge trail was',
    'muddy but passable, and the old bridge held up',
    'better than expected. Counted eleven herons near',
    'the reeds and one very curious otter.',
    '',
    'Supplies: 2 days of food, 3 L water, spare socks.',
    'Next camp: the pine grove, about 6 km north.',
  ];
  lines.forEach((t, i) => g.fillText(t, 34, 96 + i * 26));
  g.strokeStyle = 'rgba(60,50,40,0.25)';
  g.strokeRect(20, 20, W - 40, H - 40);
  const small = canvas(390, 270);
  const s = small.getContext('2d');
  s.filter = 'blur(0.7px)';
  s.drawImage(c, 0, 0, 390, 270);
  s.filter = 'none';
  addGrain(s, 390, 270, 14, 3);
  return toFile(small, 'sample-notes.jpg', 'image/jpeg', 0.5);
}

function pixelArt() {
  const art = [
    '................',
    '.....KKKKKK.....',
    '...KKYYYYYYKK...',
    '..KYYYYYYYYYYK..',
    '.KYYKKYYYYKKYYK.',
    '.KYYKKYYYYKKYYK.',
    'KYYYYYYYYYYYYYYK',
    'KYYRRYYYYYYRRYYK',
    'KYYYYYYYYYYYYYYK',
    'KYYYKYYYYYYKYYYK',
    '.KYYYKKKKKKYYYK.',
    '.KYYYYYYYYYYYYK.',
    '..KYYYYYYYYYYK..',
    '...KKYYYYYYKK...',
    '.....KKKKKK.....',
    '................',
  ];
  const colors = { K: '#2b2340', Y: '#ffd23f', R: '#ff6b8b' };
  const c = canvas(16, 16);
  const g = c.getContext('2d');
  art.forEach((row, y) => [...row].forEach((ch, x) => {
    if (colors[ch]) {
      g.fillStyle = colors[ch];
      g.fillRect(x, y, 1, 1);
    }
  }));
  return toFile(c, 'sample-pixel-smiley.png', 'image/png');
}

export const SAMPLES = [
  { id: 'sunset', label: 'Sunset photo', make: landscape },
  { id: 'notes', label: 'Scanned notes', make: documentScan },
  { id: 'pixel', label: 'Pixel art', make: pixelArt },
];
