/* Public task records only; all untrusted text is inserted with textContent. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const terminal = new Set(['succeeded', 'aborted', 'error', 'canceled', 'not_found', 'inconclusive', 'interrupted']);
  const labels = { pending: '待执行', skipped: '已跳过', planned: '等待确认', running: '执行中', canceling: '正在停止', canceled: '已取消',
    succeeded: '已完成', aborted: '执行中止', error: '执行错误', not_found: '未找到', inconclusive: '无法确认', interrupted: '已中断' };
  const stepNames = { navigate: '前往地图目标', observe: '到点观察', return: '返回出发点',
    find_object: '检查候选点', welcome: '迎宾接待', wait_guest: '等待访客确认', pickup: '到达接人点' };
  let cid = sessionStorage.getItem('jaka-replay-conversation') || '';
  let snapshot = null;
  let requestBusy = false;
  let latestPlan = null;
  let restoredCase = false;
  async function api(path, body, timeout = 10000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { method: body ? 'POST' : 'GET', signal: controller.signal,
        headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '服务暂不可用');
      return data;
    } finally { clearTimeout(timer); }
  }
  function error(message) { $('errorMessage').textContent = message || ''; $('errorMessage').hidden = !message; }
  function actionButtons() {
    const task = snapshot?.task;
    const status = task?.status;
    const owned = task?.conversation_id === cid;
    const pending = owned && task?.guest_confirmation?.status === 'pending' && status === 'running';
    $('planButton').disabled = requestBusy || ['planned', 'running', 'canceling'].includes(status);
    $('caseSelect').disabled = requestBusy || ['planned', 'running', 'canceling'].includes(status);
    $('executeButton').disabled = requestBusy || !owned || status !== 'planned';
    $('cancelButton').disabled = requestBusy || !owned || !['planned', 'running'].includes(status);
    $('feedbackButton').disabled = requestBusy || !owned || !terminal.has(status);
    $('guestActions').hidden = !pending;
    $('acceptGuestButton').disabled = requestBusy || !pending;
    $('rejectGuestButton').disabled = requestBusy || !pending;
  }
  function caseDescription() {
    const item = snapshot?.cases?.find(value => value.id === $('caseSelect').value);
    $('caseDescription').textContent = item?.description || '选择案例，查看任务从规划到反馈的完整过程。';
  }
  function timeline(id, events) {
    const list = $(id);
    const scroll = list.scrollTop;
    const atBottom = list.scrollHeight - scroll - list.clientHeight < 40;
    list.replaceChildren();
    if (!events.length) { const li = document.createElement('li'); li.className = 'empty'; li.textContent = '等待任务事件'; list.append(li); }
    events.slice(-16).forEach(event => {
      const li = document.createElement('li');
      const text = document.createElement('span'); text.textContent = event.title; li.append(text);
      const small = document.createElement('small');
      const source = event.evidence?.source_type === 'synthetic_replay' ? ' · 合成观察样例' : '';
      small.textContent = new Date(event.time * 1000).toLocaleTimeString() + source;
      li.append(small); list.append(li);
    });
    list.scrollTop = atBottom ? list.scrollHeight : scroll;
  }
  function map(graph, robot, task) {
    const svg = $('worldMap'); svg.replaceChildren();
    const point = xy => [60 + Number(xy[0]) * 65, 235 - Number(xy[1]) * 43];
    function shape(tag, attributes) {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
      svg.append(element); return element;
    }
    [0, 1, 2, 3, 4, 5, 6, 7].forEach(x => shape('line', { x1: 60 + x * 65, x2: 60 + x * 65, y1: 30, y2: 250, stroke: '#e8edf4' }));
    if (robot.track?.length) shape('polyline', { points: robot.track.map(point).map(p => p.join(',')).join(' '), fill: 'none', stroke: '#8cc9be', 'stroke-width': 3 });
    (graph.objects || []).forEach(object => {
      const [x, y] = point(object.floor_xy);
      const active = object.ann_id === task?.current_target_ann_id;
      shape('rect', { x: x - 16, y: y - 11, width: 32, height: 22, rx: 5, fill: active ? '#c5d6fa' : '#dfe7f2', stroke: active ? '#3567dc' : '#bdcbe0' });
      const text = shape('text', { x, y: y + 30, 'text-anchor': 'middle' }); text.textContent = object.category_zh || object.category;
    });
    const [x, y] = point([robot.pose?.x || 0, robot.pose?.y || 0]);
    shape('circle', { cx: x, cy: y, r: 9, fill: '#168274', stroke: 'white', 'stroke-width': 3, class: 'robot' });
    $('robotPose').textContent = `当前位置：x ${(robot.pose?.x || 0).toFixed(2)} / y ${(robot.pose?.y || 0).toFixed(2)} m`;
  }
  function render(data) {
    snapshot = data;
    if (!restoredCase) {
      const current = data.case_id || data.task?.skill?.id;
      if (data.cases.some(item => item.id === current)) $('caseSelect').value = current;
      restoredCase = true;
    }
    caseDescription();
    $('modeNotice').textContent = data.mode === 'model'
      ? '模型决策模式 · Agent 调用已配置模型；机器人动作与视觉观察使用合成回放样例。'
      : '离线回放模式 · 使用预设工具决策和合成观察样例，无需模型和机器人硬件。';
    const task = data.task;
    const reference = task?.reference?.image_url || latestPlan?.reference?.image_url;
    $('referenceBox').hidden = !reference;
    if (reference?.startsWith('/references/')) $('referenceImage').src = reference;
    $('taskStatus').textContent = labels[task?.status] || '尚未生成';
    $('planSteps').replaceChildren();
    (task?.steps || []).forEach(step => {
      const li = document.createElement('li');
      const target = Array.isArray(step.target_names) ? step.target_names.join('、') : step.target_names || '';
      li.textContent = `${stepNames[step.type] || '任务步骤'}${target ? ' · ' + target : ''}${step.status ? ' · ' + (labels[step.status] || step.status) : ''}`;
      $('planSteps').append(li);
    });
    if (!task) { const li = document.createElement('li'); li.textContent = '等待资源核对与任务规划'; $('planSteps').append(li); }
    timeline('agentEvents', data.events.filter(event => event.actor === 'agent' && event.kind === 'tool'));
    timeline('robotEvents', data.events.filter(event => event.actor !== 'agent' && event.kind !== 'request'));
    map(data.graph, data.robot, task);
    const observation = task?.observations?.at(-1);
    const imageUrl = task?.guest_confirmation?.status === 'pending' ? task.guest_confirmation.image_url : observation?.image_url;
    $('observationBox').hidden = !imageUrl;
    if (imageUrl?.startsWith('/captures/')) $('observationImage').src = imageUrl;
    $('observationText').textContent = '合成观察样例 · ' + (observation?.text || '到点反馈');
    $('executionResult').textContent = task?.result_text || task?.error || task?.current_stage || '等待任务确认与执行反馈。';
    const response = data.events.filter(event => event.kind === 'response').at(-1);
    if (response) $('agentAnswer').textContent = response.title;
    const feedback = data.events.filter(event => event.kind === 'feedback').at(-1);
    $('feedbackAnswer').textContent = feedback?.title || '';
    actionButtons();
  }
  async function action(callback) {
    if (requestBusy) return;
    requestBusy = true; error(''); actionButtons();
    try { await callback(); await refresh(); }
    catch (failure) { error(failure.name === 'AbortError' ? '请求超时，请检查任务状态后再试。' : failure.message); }
    finally { requestBusy = false; actionButtons(); }
  }
  async function refresh() {
    const requested = cid;
    const data = await api('/api/replay' + (requested ? '?conversation_id=' + encodeURIComponent(requested) : ''));
    if (requested === cid) render(data);
  }
  $('caseSelect').addEventListener('change', caseDescription);
  $('planButton').addEventListener('click', () => action(async () => {
    cid = crypto.randomUUID().replaceAll('-', ''); sessionStorage.setItem('jaka-replay-conversation', cid);
    latestPlan = null;
    $('agentAnswer').textContent = '正在查询资源并生成计划…';
    latestPlan = await api('/api/replay/plan', { case_id: $('caseSelect').value, conversation_id: cid }, 180000);
    $('agentAnswer').textContent = latestPlan.result.text;
    const reference = latestPlan.reference?.image_url;
    $('referenceBox').hidden = !reference;
    if (reference?.startsWith('/references/')) $('referenceImage').src = reference;
  }));
  $('executeButton').addEventListener('click', () => action(() => api('/api/task/execute', { task_id: snapshot.task.id, conversation_id: cid })));
  $('cancelButton').addEventListener('click', () => action(() => api('/api/task/cancel', { task_id: snapshot.task.id, conversation_id: cid })));
  $('feedbackButton').addEventListener('click', () => action(() => api('/api/replay/feedback', { conversation_id: cid }, 180000)));
  function confirmGuest(accepted) { return action(() => api('/api/task/welcome-confirm', {
    task_id: snapshot.task.id, confirmation_id: snapshot.task.guest_confirmation.id, accepted, conversation_id: cid })); }
  $('acceptGuestButton').addEventListener('click', () => confirmGuest(true));
  $('rejectGuestButton').addEventListener('click', () => confirmGuest(false));
  async function poll() {
    try { await refresh(); }
    catch (failure) { error('任务服务连接失败，请确认已使用 --replay 启动。'); }
    setTimeout(poll, document.hidden ? 2000 : 600);
  }
  poll();
})();
