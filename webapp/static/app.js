/* Power Design Toolkit - web shell.
 * Pure client: submits parameters, renders server results. No calculation here.
 */
const $ = (id) => document.getElementById(id);
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
const plotLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#aebbd0', size: 11 }, margin: { l: 58, r: 22, t: 12, b: 52 },
  xaxis: { gridcolor: '#253148', zerolinecolor: '#253148' },
  yaxis: { gridcolor: '#253148', zerolinecolor: '#253148' },
  legend: { orientation: 'h', y: -0.2 },
};
const fmt = (v, n = 2) => Number.isFinite(v) ? Number(v).toLocaleString(undefined, { minimumFractionDigits: n, maximumFractionDigits: n }) : '—';
const metric = (label, value) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;

/* ---------------- router ---------------- */
const ROUTES = {
  'llc': 'view-llc', 'control': 'view-control', 'database': 'view-database',
  'report': 'view-report', 'pfc': 'view-placeholder', 'vienna': 'view-placeholder', 'dab': 'view-placeholder',
};
const PLACEHOLDER = {
  pfc: ['PFC 设计（TTPL / Vienna）', 'PFC 内核（CCM/CRM/DCM、电感综合、控制实验）已随 Python 打包，Web 接入按路线图推进。'],
  vienna: ['Vienna 整流器', '三相 Vienna dq 控制实验面板，Web 接入按路线图推进。'],
  dab: ['DAB 双向变换器', 'TPS/ZVS 双向 DAB 设计模块，Web 接入按路线图推进。'],
};
let defaults = null;

function route() {
  const hash = location.hash.replace(/^#\//, '') || 'llc';
  const key = hash.split('/')[0];
  document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  let activeLink = null;
  document.querySelectorAll('#nav a').forEach((a) => {
    const active = a.dataset.route === key;
    a.classList.toggle('active', active);
    if (active) {
      a.setAttribute('aria-current', 'page');
      activeLink = a;
    } else {
      a.removeAttribute('aria-current');
    }
  });
  activeLink?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  const viewId = ROUTES[key] || 'view-placeholder';
  if (viewId === 'view-placeholder') {
    const [title, text] = PLACEHOLDER[key] || ['模块', '该模块尚未接入计算内核。'];
    $('phTitle').textContent = title;
    $('phText').textContent = text;
  }
  $(viewId).classList.remove('hidden');
  if (key === 'database') loadDatabase();
}
window.addEventListener('hashchange', route);

/* ---------------- LLC view ---------------- */
function setSelect(id, options, selected) {
  const s = $(id); s.innerHTML = '';
  options.forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; o.selected = v === selected; s.appendChild(o); });
}
function setValue(id, v) { $(id).value = Number.isFinite(v) ? v : ''; }
function populate(d) {
  defaults = d; const s = d.spec;
  ['vbus_nom_v', 'vbus_min_normal_v', 'vbus_max_v', 'vbus_hold_end_v', 'vout_v', 'ln_ratio', 'q_full_load', 'primary_turns', 'secondary_turns'].forEach((k) => setValue(k, s[k]));
  setValue('pout_kw', s.pout_w / 1000); setValue('fr_khz', s.resonant_frequency_hz / 1000);
  setValue('fmin_khz', s.minimum_frequency_hz / 1000); setValue('fmax_khz', s.maximum_frequency_hz / 1000);
  setValue('primary_deadtime_ns', s.primary_deadtime_s * 1e9); setValue('bus_cap_uf', s.bus_capacitance_f * 1e6);
  setSelect('primary_topology', d.topologies, s.primary_topology);
  setSelect('primary_device', d.primary_devices, s.primary_device);
  setSelect('sr_device', d.sr_devices, s.sr_device);
}
function specFromForm() {
  const val = (id) => Number($(id).value);
  return {
    vbus_nom_v: val('vbus_nom_v'), vbus_min_normal_v: val('vbus_min_normal_v'), vbus_max_v: val('vbus_max_v'),
    vbus_hold_end_v: val('vbus_hold_end_v'), vout_v: val('vout_v'), pout_w: val('pout_kw') * 1000,
    resonant_frequency_hz: val('fr_khz') * 1000, minimum_frequency_hz: val('fmin_khz') * 1000,
    maximum_frequency_hz: val('fmax_khz') * 1000, ln_ratio: val('ln_ratio'), q_full_load: val('q_full_load'),
    primary_turns: val('primary_turns'), secondary_turns: val('secondary_turns'),
    primary_topology: $('primary_topology').value, primary_device: $('primary_device').value,
    sr_device: $('sr_device').value, primary_deadtime_s: val('primary_deadtime_ns') * 1e-9,
    bus_capacitance_f: val('bus_cap_uf') * 1e-6,
  };
}
function renderSummary(r) {
  const s = r.summary;
  $('statusBadge').textContent = r.status;
  $('statusBadge').className = `status ${r.feasible ? 'pass' : 'fail'}`;
  $('statusMessage').textContent = r.feasible ? '设计满足当前模型约束' : '存在不可行约束';
  $('heroMetrics').innerHTML =
    `<div><span>Nominal η</span><strong>${fmt(s.nominal_efficiency * 100, 3)}%</strong></div>` +
    `<div><span>Nominal fs</span><strong>${fmt(s.nominal_switching_frequency_hz / 1000, 2)} kHz</strong></div>` +
    `<div><span>ZVS margin</span><strong>${fmt(s.nominal_zvs_margin, 2)}×</strong></div>`;
  $('metrics').innerHTML = [
    metric('Lr', `${fmt(s.lr_h * 1e6, 3)} µH`), metric('Cr', `${fmt(s.cr_f * 1e9, 3)} nF`),
    metric('Lm', `${fmt(s.lm_h * 1e6, 2)} µH`), metric('Turns ratio', fmt(s.turns_ratio, 3)),
    metric('Transformer', s.transformer_core), metric('Resonant L', `${s.resonant_inductor_core} · ${s.resonant_inductor_turns}T`),
  ].join('');
  // right result rail
  const rail = $('railStatus');
  rail.textContent = r.status; rail.className = `rail-status ${r.feasible ? 'pass' : 'fail'}`;
  $('railEff').textContent = `${fmt(s.nominal_efficiency * 100, 3)}%`;
  $('railTemp').textContent = s.max_hotspot_c != null ? `${fmt(s.max_hotspot_c, 1)} ℃` : '—';
  $('railZvs').textContent = s.nominal_zvs_margin >= s.zvs_required_margin ? 'YES' : 'NO';
  $('railFreq').textContent = `${fmt(s.nominal_switching_frequency_hz / 1000, 1)} kHz`;
  $('railXfrm').textContent = s.transformer_core;
  $('railCost').textContent = s.bom_cost_usd != null ? `$${fmt(s.bom_cost_usd, 2)}` : '—';
}
function renderGain(r) {
  const g = r.gain_map;
  const traces = g.curves.map((c) => ({
    x: c.frequency_hz.map((x) => x / 1000), y: c.gain, type: 'scatter', mode: 'lines',
    name: `${Math.round(c.load_fraction * 100)}% load`, line: { width: 2 },
  }));
  g.targets.forEach((t) => traces.push({
    x: [g.curves[0].frequency_hz[0] / 1000, g.curves[0].frequency_hz.at(-1) / 1000],
    y: [t.gain, t.gain], type: 'scatter', mode: 'lines', name: `Target ${t.vbus_v.toFixed(0)}V`,
    line: { dash: 'dot', width: 1 },
  }));
  traces.push({
    x: r.operating_points.map((p) => p.switching_frequency_hz / 1000),
    y: r.operating_points.map((p) => p.achieved_gain),
    text: r.operating_points.map((p) => p.label), type: 'scatter', mode: 'markers',
    name: 'Operating points', marker: { size: 9, symbol: 'diamond' },
  });
  Plotly.react('gainChart', traces, {
    ...plotLayout, xaxis: { ...plotLayout.xaxis, title: 'Switching frequency / kHz' },
    yaxis: { ...plotLayout.yaxis, title: 'Normalized gain' },
  }, plotConfig);
}
function renderLoss(r) {
  const labels = {
    primary_conduction_w: 'Primary cond.', primary_turnoff_w: 'Primary Eoff', primary_gate_w: 'Primary gate',
    primary_coss_w: 'Primary Coss', primary_deadtime_w: 'Primary DT', sr_conduction_w: 'SR cond.',
    sr_deadtime_w: 'SR DT', sr_turnoff_w: 'SR Eoff', sr_coss_w: 'SR Coss', sr_gate_w: 'SR gate',
    transformer_core_w: 'XFMR core', transformer_primary_copper_w: 'XFMR pri Cu',
    transformer_secondary_copper_w: 'XFMR sec Cu', resonant_inductor_core_w: 'Lr core',
    resonant_inductor_copper_w: 'Lr Cu', resonant_capacitor_w: 'Cr', output_capacitor_w: 'Co', auxiliary_w: 'Aux',
  };
  const rows = Object.entries(r.nominal_loss_breakdown).filter((x) => x[1] > 0.01).sort((a, b) => a[1] - b[1]);
  Plotly.react('lossChart', [{
    x: rows.map((x) => x[1]), y: rows.map((x) => labels[x[0]] || x[0]), type: 'bar', orientation: 'h',
    text: rows.map((x) => fmt(x[1], 2)), textposition: 'auto',
  }], {
    ...plotLayout, margin: { l: 105, r: 18, t: 10, b: 45 }, showlegend: false,
    xaxis: { ...plotLayout.xaxis, title: 'Loss / W' },
  }, plotConfig);
}
function renderEfficiency(r) {
  const p = r.operating_points;
  Plotly.react('effChart', [{
    x: p.map((x) => x.label), y: p.map((x) => x.efficiency * 100), type: 'bar',
    text: p.map((x) => fmt(x.efficiency * 100, 2) + '%'), textposition: 'auto',
  }], {
    ...plotLayout, showlegend: false, margin: { l: 55, r: 18, t: 10, b: 80 },
    xaxis: { ...plotLayout.xaxis, tickangle: -35 },
    yaxis: { ...plotLayout.yaxis, title: 'Efficiency / %', range: [Math.max(0, Math.min(...p.map((x) => x.efficiency * 100)) - 2), 100] },
  }, plotConfig);
}
function renderTable(r) {
  $('opTable').innerHTML = r.operating_points.map((p) =>
    `<tr><td>${p.label}</td><td>${fmt(p.switching_frequency_hz / 1000, 2)}</td><td>${fmt(p.input_phase_deg, 2)}</td>` +
    `<td>${fmt(p.resonant_current_rms_a, 2)}</td><td>${fmt(p.zvs_margin, 2)}</td><td>${fmt(p.total_loss_w, 2)}</td>` +
    `<td>${fmt(p.efficiency * 100, 3)}</td><td>${p.branch}</td></tr>`).join('');
}
function renderMessages(r) {
  const all = [...r.feasibility_reasons.map((x) => ({ x, err: true })), ...r.warnings.map((x) => ({ x, err: false }))];
  const p = $('messagesPanel');
  if (!all.length) { p.classList.add('hidden'); return; }
  p.classList.remove('hidden');
  $('messages').innerHTML = all.map((m) => `<div class="msg ${m.err ? 'error' : ''}">${m.x}</div>`).join('');
}
function render(r) {
  const steps = [
    ['Summary', renderSummary], ['Gain', renderGain], ['Loss', renderLoss],
    ['Efficiency', renderEfficiency], ['Table', renderTable], ['Messages', renderMessages],
  ];
  for (const [name, fn] of steps) {
    try { fn(r); } catch (e) { console.error(`render ${name} failed:`, e); }
  }
}

async function runAnalysis() {
  $('runBtn').disabled = true; $('runBtn').textContent = '服务器计算中…';
  $('errorBox').classList.add('hidden');
  $('statusBadge').textContent = 'RUN'; $('statusBadge').className = 'status pending';
  $('statusMessage').textContent = 'Python 模型正在计算';
  try {
    const res = await fetch('/api/llc/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec: specFromForm() }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    render(body);
  } catch (e) {
    $('errorBox').textContent = e.message; $('errorBox').classList.remove('hidden');
    $('statusBadge').textContent = 'ERR'; $('statusBadge').className = 'status fail';
    $('statusMessage').textContent = '计算失败';
  } finally {
    $('runBtn').disabled = false; $('runBtn').textContent = '运行完整 LLC 计算';
  }
}

/* ---------------- Control Loop view ---------------- */
function cPlantContext() {
  return {
    plant: {
      lr_h: Number($('c_lr').value) * 1e-6, cr_f: Number($('c_cr').value) * 1e-9,
      lm_h: Number($('c_lm').value) * 1e-6, turns_ratio: Number($('c_ratio').value),
      vbus_v: Number($('c_vin').value), vout_v: Number($('c_vout').value), pout_w: Number($('c_pout').value),
    },
    load_fraction: Number($('c_load').value),
    sample_time_s: 1.0 / (Number($('c_fs').value) * 1000),
  };
}
function cPost(path, body) {
  return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(async (res) => { const data = await res.json(); if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`); return data; });
}
function cSwitchTab(name) {
  document.querySelectorAll('#cTabs .tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('#cTabs ~ .tab-body').forEach((b) => b.classList.add('hidden'));
  $(`cTab-${name}`).classList.remove('hidden');
}
document.querySelectorAll('#cTabs .tab').forEach((b) => b.addEventListener('click', () => cSwitchTab(b.dataset.tab)));

function renderBode(body) {
  const f = body.frequencies_hz.map((x) => x / 1000);
  const ol = body.responses.open_loop_nominal;
  const ann = body.annotations;
  const marker = ann.crossover_hz != null ? [{
    x: [ann.crossover_hz / 1000], y: [0], type: 'scatter', mode: 'markers',
    marker: { size: 10, symbol: 'star', color: '#ffd166' },
    name: `Crossover ${fmt(ann.crossover_hz, 0)} Hz`,
  }] : [];
  Plotly.react('bodeMag', [
    { x: f, y: ol.magnitude_db, type: 'scatter', mode: 'lines', name: 'Open loop', line: { width: 2 } },
    { x: f, y: Array(f.length).fill(0), type: 'scatter', mode: 'lines', name: '0 dB', line: { dash: 'dot', color: '#7ee787' } },
    ...marker,
  ], {
    ...plotLayout, showlegend: true, xaxis: { ...plotLayout.xaxis, title: 'Frequency / kHz', type: 'log' },
    yaxis: { ...plotLayout.yaxis, title: 'Magnitude / dB' },
  }, plotConfig);
  const pmMarker = ann.phase_margin_deg != null && ann.crossover_hz != null ? [{
    x: [ann.crossover_hz / 1000], y: [-180 + ann.phase_margin_deg], type: 'scatter', mode: 'markers',
    marker: { size: 10, symbol: 'star', color: '#ffd166' },
    name: `PM ${fmt(ann.phase_margin_deg, 1)}°`,
  }] : [];
  Plotly.react('bodePhase', [
    { x: f, y: ol.phase_deg, type: 'scatter', mode: 'lines', name: 'Phase', line: { width: 2 } },
    { x: f, y: Array(f.length).fill(-180), type: 'scatter', mode: 'lines', name: '-180°', line: { dash: 'dot', color: '#ff7b72' } },
    ...pmMarker,
  ], {
    ...plotLayout, showlegend: true, xaxis: { ...plotLayout.xaxis, title: 'Frequency / kHz', type: 'log' },
    yaxis: { ...plotLayout.yaxis, title: 'Phase / deg' },
  }, plotConfig);
}
function renderLoop(loop) {
  const rows = [
    ['标称延迟', loop.margins], ['最小延迟', loop.margins_min_delay], ['最大延迟', loop.margins_max_delay],
  ];
  $('cLoopTable').innerHTML = rows.map(([label, m]) =>
    `<tr><td>${label}</td><td>${fmt(m.crossover_hz, 0)}</td><td>${fmt(m.phase_margin_deg, 1)}</td>` +
    `<td>${fmt(m.gain_margin_db, 1)}</td><td>${m.stable ? 'PASS' : 'FAIL'}</td></tr>`).join('');
}
function renderRail(design) {
  const m = design.achieved || {};
  const rail = $('cRailStatus');
  rail.textContent = m.stable ? 'PASS' : 'FAIL';
  rail.className = `rail-status ${m.stable ? 'pass' : 'fail'}`;
  $('cRailBw').textContent = m.crossover_hz != null ? `${fmt(m.crossover_hz, 0)} Hz` : '—';
  $('cRailPm').textContent = m.phase_margin_deg != null ? `${fmt(m.phase_margin_deg, 1)}°` : '—';
  $('cRailGm').textContent = m.gain_margin_db != null ? `${fmt(m.gain_margin_db, 1)} dB` : '—';
  $('cRailType').textContent = design.type || '—';
  $('cRailTarget').textContent = `${fmt(Number($('c_bw').value), 0)} Hz / ${fmt(Number($('c_pm').value), 0)}°`;
  const badge = $('cStatusBadge');
  badge.textContent = m.stable ? 'PASS' : 'FAIL';
  badge.className = `status ${m.stable ? 'pass' : 'fail'}`;
  $('cStatusMessage').textContent = m.stable ? '环路稳定' : '环路不稳定';
  $('cHeroMetrics').innerHTML =
    `<div><span>Bandwidth</span><strong>${$('cRailBw').textContent}</strong></div>` +
    `<div><span>Phase Margin</span><strong>${$('cRailPm').textContent}</strong></div>` +
    `<div><span>Gain Margin</span><strong>${$('cRailGm').textContent}</strong></div>`;
}
function renderComp(design) {
  const c = design.continuous, d = design.discrete, m = design.achieved;
  $('cCompMetrics').innerHTML = [
    metric('类型', design.type), metric('迭代', `${design.iterations}`),
    metric('Bandwidth', m.crossover_hz != null ? `${fmt(m.crossover_hz, 0)} Hz` : '—'),
    metric('Phase Margin', m.phase_margin_deg != null ? `${fmt(m.phase_margin_deg, 1)}°` : '—'),
    metric('Gain Margin', m.gain_margin_db != null ? `${fmt(m.gain_margin_db, 1)} dB` : '—'),
    metric('Stable', m.stable ? 'PASS' : 'FAIL'),
  ].join('');
  const analogRows = [
    ['Kp / Kc', fmt(c.Kp ?? c.Kc, 5)], ['Ki', fmt(c.Ki, 4)],
    ['Zeros / Hz', (c.zeros_hz || []).map((z) => fmt(z, 1)).join(', ')],
    ['Poles / Hz', (c.poles_hz || []).map((z) => fmt(z, 1)).join(', ')],
  ];
  const digitalRows = [
    ['b0', fmt(d.b0, 6)], ['b1', fmt(d.b1, 6)], ['b2', fmt(d.b2, 6)], ['a1', fmt(d.a1, 6)], ['a2', fmt(d.a2, 6)],
  ];
  $('cCompTable').innerHTML = analogRows.map(([k, v], i) =>
    `<tr><td>${k}</td><td>${v}</td><td>${digitalRows[i] ? digitalRows[i][0] : ''}</td><td>${digitalRows[i] ? digitalRows[i][1] : ''}</td></tr>`).join('');
}
function renderPlant(body) {
  const tf = body.continuous;
  $('cPlantMetrics').innerHTML = [
    metric('输入', tf.input_name), metric('输出', tf.output_name),
    metric('DC 增益', fmt(tf.dc_gain, 5)), metric('fsw', `${fmt(body.operating_point.switching_frequency_hz / 1000, 2)} kHz`),
    metric('FM 增益', fmt(body.fm_gain_hz_per_pu, 0)), metric('采样', `${fmt(1 / body.sample_time_s / 1000, 0)} kHz`),
  ].join('');
  const n = Math.max(tf.poles.length, tf.zeros.length);
  let rows = '';
  for (let i = 0; i < n; i++) {
    const p = tf.poles[i], z = tf.zeros[i];
    rows += `<tr><td>${p ? `${fmt(p[0], 1)} ${p[1] ? '± j' + fmt(Math.abs(p[1]), 1) : ''}` : ''}</td>` +
      `<td>${z ? `${fmt(z[0], 1)} ${z[1] ? '± j' + fmt(Math.abs(z[1]), 1) : ''}` : ''}</td></tr>`;
  }
  $('cPoleTable').innerHTML = rows;
  $('cPlantTf').textContent =
    `Gvd(s) = ${tf.numerator.map((c) => fmt(c, 4)).join(', ')} / ${tf.denominator.map((c) => fmt(c, 4)).join(', ')}\n` +
    `离散 ZOH: ${body.discrete_numerator.map((c) => fmt(c, 4)).join(', ')} / ${body.discrete_denominator.map((c) => fmt(c, 4)).join(', ')}`;
}
function renderDigital(body) {
  $('cDigMetrics').innerHTML = [
    metric('类型', body.kind), metric('采样', `${fmt(1 / body.sample_time_s / 1000, 0)} kHz`),
  ].join('');
  $('cDigCode').textContent = `${body.difference_equation}\n\n` + body.c_code;
}
function renderProtection(body) {
  const t = body.thresholds;
  $('cProtMetrics').innerHTML = [
    metric('OVP', `${fmt(t.ovp_v, 2)} V`), metric('UVP', `${fmt(t.uvp_v, 2)} V`),
    metric('OCP', `${fmt(t.ocp_a, 2)} A`), metric('频率钳位', `${fmt(t.freq_min_hz / 1000, 0)}–${fmt(t.freq_max_hz / 1000, 0)} kHz`),
    metric('软启动', `${fmt(t.soft_start_start_hz / 1000, 0)} kHz`), metric('Burst', `${fmt(t.burst_fraction * 100, 0)}%`),
  ].join('');
  $('cProtTable').innerHTML = body.states.map((s) =>
    `<tr><td>${s.name}</td><td>${s.description}</td><td>${s.actions.join('；')}</td></tr>`).join('');
  $('cTransTable').innerHTML = body.transitions.map((tr) =>
    `<tr><td>${tr.source}</td><td>→</td><td>${tr.condition}</td></tr>`).join('');
}
async function cRun() {
  $('cRunBtn').disabled = true; $('cRunBtn').textContent = '服务器计算中…';
  $('cErrorBox').classList.add('hidden');
  $('cStatusBadge').textContent = 'RUN'; $('cStatusBadge').className = 'status pending';
  $('cStatusMessage').textContent = '内核环路分析中（数秒）';
  try {
    const ctx = cPlantContext();
    // 1) design compensator (kernel-margin loop) -> margins + coefficients
    const design = await cPost('/api/control/compensator', {
      plant_context: ctx, type: $('c_type').value,
      target_bandwidth_hz: Number($('c_bw').value), target_phase_margin_deg: Number($('c_pm').value),
    });
    // 2) bode with the designed 2p2z
    const bode = await cPost('/api/control/bode', {
      plant_context: ctx,
      controller: { kind: '2p2z', b0: design.discrete.b0, b1: design.discrete.b1, b2: design.discrete.b2, a1: design.discrete.a1, a2: design.discrete.a2 },
    });
    renderComp(design); renderBode(bode); renderRail(design);
    $('cLoopTable').innerHTML = '';
    cSwitchTab('comp');
    // 3) background: plant / loop / digital / protection data
    cPost('/api/control/plant', ctx).then(renderPlant).catch(() => {});
    cPost('/api/control/loop_gain', {
      plant_context: ctx,
      controller: { kind: '2p2z', b0: design.discrete.b0, b1: design.discrete.b1, b2: design.discrete.b2, a1: design.discrete.a1, a2: design.discrete.a2 },
    }).then(renderLoop).catch(() => {});
    cPost('/api/control/digital_controller', {
      plant_context: ctx,
      controller: { kind: '2p2z', b0: design.discrete.b0, b1: design.discrete.b1, b2: design.discrete.b2, a1: design.discrete.a1, a2: design.discrete.a2 },
    }).then(renderDigital).catch(() => {});
    cPost('/api/control/protection', ctx).then(renderProtection).catch(() => {});
  } catch (e) {
    $('cErrorBox').textContent = e.message; $('cErrorBox').classList.remove('hidden');
    $('cStatusBadge').textContent = 'ERR'; $('cStatusBadge').className = 'status fail';
    $('cStatusMessage').textContent = '分析失败';
  } finally {
    $('cRunBtn').disabled = false; $('cRunBtn').textContent = '设计补偿器并分析';
  }
}
async function cPlantOnly() {
  $('cErrorBox').classList.add('hidden');
  try {
    const body = await cPost('/api/control/plant', cPlantContext());
    renderPlant(body);
    cSwitchTab('plant');
  } catch (e) {
    $('cErrorBox').textContent = e.message; $('cErrorBox').classList.remove('hidden');
  }
}

/* ---------------- Database view ---------------- */
async function loadDatabase() {
  if ($('dbTable').dataset.loaded) return;
  try {
    const d = await fetch('/api/llc/cores').then((r) => r.json());
    const familySel = $('dbFamily');
    d.families.forEach((f) => { const o = document.createElement('option'); o.value = f; o.textContent = f; familySel.appendChild(o); });
    const render = () => {
      const fam = familySel.value;
      $('dbTable').innerHTML = d.cores.filter((c) => !fam || c.family === fam).map((c) =>
        `<tr><td>${c.part_number}</td><td>${c.family}</td><td>${c.shape}</td><td>${c.material_key}</td>` +
        `<td>${fmt(c.ae_mm2, 1)}</td><td>${fmt(c.ve_mm3, 0)}</td><td>${fmt(c.le_mm, 1)}</td>` +
        `<td>${fmt(c.core_mass_g, 1)}</td><td>${fmt(c.cost_usd, 2)}</td><td>${fmt(c.thermal_resistance_k_per_w, 1)}</td></tr>`).join('');
    };
    familySel.addEventListener('change', render);
    render();
    $('dbTable').dataset.loaded = '1';
  } catch (e) {
    $('dbTable').innerHTML = `<tr><td colspan="10">${e.message}</td></tr>`;
  }
}

/* ---------------- Report view ---------------- */
function currentSpec() { return defaults ? specFromForm() : {}; }
async function downloadReport(path, filename, body) {
  const res = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
$('rPdfBtn').addEventListener('click', async () => {
  $('rErrorBox').classList.add('hidden');
  try {
    await downloadReport('/api/llc/report', 'llc_design_report.pdf', {
      spec: currentSpec(), project: $('r_project').value, engineer: $('r_engineer').value,
    });
  } catch (e) { $('rErrorBox').textContent = e.message; $('rErrorBox').classList.remove('hidden'); }
});
$('rXlsxBtn').addEventListener('click', async () => {
  $('rErrorBox').classList.add('hidden');
  try {
    await downloadReport('/api/llc/report.xlsx', 'llc_design_report.xlsx', { spec: currentSpec() });
  } catch (e) { $('rErrorBox').textContent = e.message; $('rErrorBox').classList.remove('hidden'); }
});

/* ---------------- boot ---------------- */
$('runBtn').addEventListener('click', runAnalysis);
$('resetBtn').addEventListener('click', () => { if (defaults) { populate(defaults); runAnalysis(); } });
$('cRunBtn').addEventListener('click', cRun);
$('cPlantBtn').addEventListener('click', cPlantOnly);
$('railReport').addEventListener('click', () => { location.hash = '#/report'; });
$('cReportBtn').addEventListener('click', () => { location.hash = '#/report'; });

async function boot() {
  try {
    const healthResponse = await fetch('/api/health');
    if (!healthResponse.ok) throw new Error(`HTTP ${healthResponse.status}`);
    const health = await healthResponse.json();
    if (health.status !== 'ok') throw new Error('unexpected health response');
    $('healthDot').style.background = '#7ee787';
    $('healthText').textContent = 'Python 服务在线';
  } catch (e) {
    $('healthDot').style.background = '#ff7b72';
    $('healthText').textContent = '服务器离线';
  }
  route();
  try {
    const d = await fetch('/api/llc/defaults').then((r) => r.json());
    populate(d);
    await runAnalysis();
  } catch (e) {
    $('errorBox').textContent = e.message;
    $('errorBox').classList.remove('hidden');
  }
}
boot();
