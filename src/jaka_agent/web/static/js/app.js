(() => {
  const STORAGE_KEY = 'jaka-vision-conversations-v1';
  const AUTO_SEND_VOICE_KEY = 'jaka-vision-auto-send-voice';
  const MAX_RECORDING_MS = 30000;
  const MAX_AUDIO_BYTES = 8 * 1024 * 1024;
  const DEFAULT_MAP_ZOOM = 1.8;
  const app = document.getElementById('app');
  const historyList = document.getElementById('historyList');
  const messageList = document.getElementById('messageList');
  const emptyState = document.getElementById('emptyState');
  const viewport = document.getElementById('messageViewport');
  const titleNode = document.getElementById('conversationTitle');
  const input = document.getElementById('promptInput');
  const composer = document.getElementById('composer');
  const sendButton = document.getElementById('sendButton');
  const micButton = document.getElementById('micButton');
  const cancelRecordingButton = document.getElementById('cancelRecordingButton');
  const autoSendVoice = document.getElementById('autoSendVoice');
  const referenceButton = document.getElementById('referenceButton');
  const referenceFileInput = document.getElementById('referenceFileInput');
  const referencePreview = document.getElementById('referencePreview');
  const referencePreviewImage = document.getElementById('referencePreviewImage');
  const referencePreviewName = document.getElementById('referencePreviewName');
  const removeReferenceButton = document.getElementById('removeReferenceButton');
  const welcomeSetup = document.getElementById('welcomeSetup');
  const welcomePickupText = document.getElementById('welcomePickupText');
  const welcomeReturnText = document.getElementById('welcomeReturnText');
  const welcomeSelectionHint = document.getElementById('welcomeSelectionHint');
  const selectWelcomePickupButton = document.getElementById('selectWelcomePickupButton');
  const selectWelcomeReturnButton = document.getElementById('selectWelcomeReturnButton');
  const composerStatus = document.getElementById('composerStatus');
  const capabilitySelect = document.getElementById('capabilitySelect');
  const capabilityHint = document.getElementById('capabilityHint');
  const patrolSetup = document.getElementById('patrolSetup');
  const patrolSelectedList = document.getElementById('patrolSelectedList');
  const patrolSelectionCount = document.getElementById('patrolSelectionCount');
  const patrolSelectionHint = document.getElementById('patrolSelectionHint');
  const patrolRounds = document.getElementById('patrolRounds');
  const planSelectedPatrol = document.getElementById('planSelectedPatrol');
  let patrolSelectedIds = [];
  let patrolMapSnapshot = null;
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  const imageDialog = document.getElementById('imageDialog');
  const dialogImage = document.getElementById('dialogImage');
  const connectionDot = document.getElementById('connectionDot');
  const connectionText = document.getElementById('connectionText');
  const mobileConnectionDot = document.getElementById('mobileConnectionDot');
  const mobileConnectionText = document.getElementById('mobileConnectionText');
  const sidebarHeading = document.getElementById('sidebarHeading');
  const workspaceSwitch = document.getElementById('workspaceSwitch');
  const mapPanel = document.getElementById('mapPanel');
  const mapResizer = document.getElementById('mapResizer');
  const mapStage = document.getElementById('mapStage');
  const mapSelect = document.getElementById('mapSelect');
  const mapFileInput = document.getElementById('mapFileInput');
  const slamFileInput = document.getElementById('slamFileInput');
  const slamVisibleInput = document.getElementById('slamVisibleInput');
  const slamName = document.getElementById('slamName');
  const slamCalibrationPanel = document.getElementById('slamCalibrationPanel');
  const slamOriginX = document.getElementById('slamOriginX');
  const slamOriginY = document.getElementById('slamOriginY');
  const slamResolution = document.getElementById('slamResolution');
  const slamRotation = document.getElementById('slamRotation');
  const slamOpacity = document.getElementById('slamOpacity');
  const mapSvg = document.getElementById('mapSvg');
  const mapLoading = document.getElementById('mapLoading');
  const mapLegend = document.getElementById('mapLegend');
  const robotStateText = document.getElementById('robotStateText');
  const objectInspector = document.getElementById('objectInspector');
  const objectInspectorMeta = document.getElementById('objectInspectorMeta');
  const scene3dStage = document.getElementById('scene3dStage');
  const scene3dCanvas = document.getElementById('scene3dCanvas');
  const scene3dStatus = document.getElementById('scene3dStatus');
  const scene3dInfo = document.getElementById('scene3dInfo');
  const scene3dInfoResizer = document.getElementById('scene3dInfoResizer');
  const realView3dButton = document.getElementById('realView3dButton');
  const reset3dViewButton = document.getElementById('reset3dViewButton');
  const currentTrackVisibleInput = document.getElementById('currentTrackVisibleInput');
  const trackHistoryEmpty = document.getElementById('trackHistoryEmpty');
  const trackHistoryList = document.getElementById('trackHistoryList');
  const mapTrackPanel = document.getElementById('mapTrackPanel');
  const trackPanelToggleButton = document.getElementById('trackPanelToggleButton');
  const mappingMain = document.getElementById('mappingMain');
  const mappingHeaderStatus = document.getElementById('mappingHeaderStatus');
  const mappingRobotState = document.getElementById('mappingRobotState');
  const mappingMaxPoints = document.getElementById('mappingMaxPoints');
  const mappingSpacing = document.getElementById('mappingSpacing');
  const mappingHeadingStep = document.getElementById('mappingHeadingStep');
  const mappingRobotRadius = document.getElementById('mappingRobotRadius');
  const mappingSafetyMargin = document.getElementById('mappingSafetyMargin');
  const mappingSettleTime = document.getElementById('mappingSettleTime');
  const mappingPlanButton = document.getElementById('mappingPlanButton');
  const mappingStartButton = document.getElementById('mappingStartButton');
  const mappingResumeButton = document.getElementById('mappingResumeButton');
  const mappingStopButton = document.getElementById('mappingStopButton');
  const mappingProgressText = document.getElementById('mappingProgressText');
  const mappingProgressBar = document.getElementById('mappingProgressBar');
  const mappingMapStage = document.getElementById('mappingMapStage');
  const mappingMapSvg = document.getElementById('mappingMapSvg');
  const mappingLatestPreviewButton = document.getElementById('mappingLatestPreviewButton');
  const mappingLatestPreview = document.getElementById('mappingLatestPreview');
  const mappingPreviewEmpty = document.getElementById('mappingPreviewEmpty');
  const mappingPreviewStatus = document.getElementById('mappingPreviewStatus');
  const mappingMetrics = document.getElementById('mappingMetrics');
  const mappingPreviewCount = document.getElementById('mappingPreviewCount');
  const mappingPreviewList = document.getElementById('mappingPreviewList');
  const mappingResultPanel = document.getElementById('mappingResultPanel');
  const mappingResultStats = document.getElementById('mappingResultStats');
  const mappingApplyButton = document.getElementById('mappingApplyButton');

  let conversations = loadConversations();
  let conversationStoreReady = false;
  let conversationStoreLoading = false;
  let pendingTurn = null;
  const composerDrafts = new Map();
  let workspaceMode = localStorage.getItem('jaka-vision-workspace') === 'mapping' ? 'mapping' : 'function';
  let activeId = localStorage.getItem(STORAGE_KEY + '-active');
  let sending = false;
  let speechState = 'idle';
  let backendOnline = false;
  let mediaRecorder = null;
  let microphoneStream = null;
  let recordingChunks = [];
  let recordingMimeType = '';
  let recordingStartedAt = 0;
  let recordingTimer = null;
  let recordingLimitTimer = null;
  let recordingCancelled = false;
  let mapData = null;
  let slamData = null;
  let slamVisible = true;
  let robotData = null;
  let activeTask = null;
  let selectedObjectId = null;
  let mapQueryIds = new Set();
  let mapZoom = DEFAULT_MAP_ZOOM;
  let mapRotation = Number(localStorage.getItem('jaka-vision-map-rotation')) || 0;
  let mapPanX = 0;
  let mapPanY = 0;
  let mapDrag = null;
  let mapVisibleWorld = { x: 1, y: 1 };
  let suppressMapClick = false;
  let pendingReference = null;
  let welcomePickupAnnId = null;
  let welcomeReturnAnnId = null;
  let welcomeSelectionRole = null;
  let scene3d = null;
  let mappingSessions = [];
  let selectedMappingId = null;
  let selectedMapping = null;
  let selectedPreviewName = null;
  autoSendVoice.checked = localStorage.getItem(AUTO_SEND_VOICE_KEY) === 'true';
  const savedSceneInfoWidth = Number(localStorage.getItem('jaka-vision-scene-info-width'));
  if (Number.isFinite(savedSceneInfoWidth) && savedSceneInfoWidth >= 150) {
    scene3dInfoResizer.parentElement.style.setProperty('--scene-info-width', `${savedSceneInfoWidth}px`);
  }
  const categoryPalette = ['#4f8fc9', '#43a267', '#e59b38', '#d56b6b', '#7a70c9', '#45a99a', '#ba6b9f', '#788792'];
  const savedTrackPalette = ['#d97706', '#2563eb', '#dc2626', '#7c3aed', '#0891b2', '#65a30d', '#c2410c', '#be185d'];
  const capabilityExamples = {
    vision: { prompt: '描述一下当前画面里有什么', hint: '拍摄当前画面并进行视觉问答' },
    map_query: { prompt: '整张拟物体地图中有没有[物体]？分别在哪里？', hint: '查询 JSON 中的全部已知物体，不拍照' },
    navigate: { prompt: '前往[目标物体]附近', hint: '规划并执行到指定地图物体的导航' },
    navigate_observe: { prompt: '前往[目标物体]附近进行环境观察并汇报现场情况', hint: '导航到目标、拍照并分析现场' },
    patrol: { prompt: '', hint: '在地图上点击需要巡逻的物体图标，再点击“生成巡逻计划”' },
    find_object: { prompt: '根据参考图片寻找这件物品', hint: '还需要点击图片按钮上传参考图' },
    welcome: { prompt: '帮我去接人点那边接一下人，如果没找到，就在那里一直等。等到看到他的时候带他回来。', hint: '上传人物上半身照片，并在地图中选择接人点和返回点' },
    return: { prompt: '前往[目标物体]附近观察，然后返回出发点', hint: '在同一个任务中完成目标操作后返回起点' },
    cancel: { prompt: '停止当前任务', hint: '取消正在执行的导航或巡逻任务' },
  };
  let currentTrackVisible = localStorage.getItem('jaka-vision-current-track-visible') !== 'false';
  let savedTracks = [];
  let visibleTrackIds = new Set();
  try {
    const storedTrackIds = JSON.parse(localStorage.getItem('jaka-vision-visible-tracks') || '[]');
    if (Array.isArray(storedTrackIds)) visibleTrackIds = new Set(storedTrackIds.map(String));
  } catch (_) {
    visibleTrackIds = new Set();
  }
  slamVisibleInput.checked = slamVisible;
  currentTrackVisibleInput.checked = currentTrackVisible;

  const savedMapWidth = Number(localStorage.getItem('jaka-vision-map-width'));
  if (Number.isFinite(savedMapWidth) && savedMapWidth >= 320) {
    const availableMapWidth = Math.max(320, window.innerWidth - 264 - 520);
    document.documentElement.style.setProperty('--map-width', `${Math.min(savedMapWidth, availableMapWidth)}px`);
  }

  function uid() {
    return (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
  }

  function loadConversations() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      if (!Array.isArray(value)) return [];
      for (const conversation of value) {
        for (const message of conversation.messages || []) {
          if (message.referenceStoredImage && (!message.referenceImage || String(message.referenceImage).startsWith('blob:'))) {
            message.referenceImage = message.referenceStoredImage;
          }
        }
      }
      return value;
    } catch (_) {
      return [];
    }
  }

  function saveConversations() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations, function(key, value) {
      if (key === 'referenceImage' && String(value || '').startsWith('blob:')) return this.referenceStoredImage || '';
      return value;
    }));
    localStorage.setItem(STORAGE_KEY + '-active', activeId || '');
  }

  function currentConversation() {
    return conversations.find(item => item.id === activeId) || null;
  }

  async function loadServerConversation(id) {
    const result = await api(`/api/conversation?id=${encodeURIComponent(id)}`);
    const index = conversations.findIndex(item => item.id === id);
    if (pendingTurn?.conversationId === id && index >= 0) {
      // Switching away/back during upload must not erase an unsent in-flight draft.
      const ids = new Set([pendingTurn.userId, pendingTurn.assistantId]);
      for (const draft of conversations[index].messages.filter(m => ids.has(m.id))) {
        const savedIndex = result.conversation.messages.findIndex(m => m.id === draft.id);
        if (savedIndex < 0) result.conversation.messages.push(draft);
        else if (result.conversation.messages[savedIndex].state === 'processing') result.conversation.messages[savedIndex] = draft;
      }
    }
    if (index >= 0) conversations[index] = result.conversation;
    else conversations.unshift(result.conversation);
    if (activeId === id) {
      activeTask = [...result.conversation.messages].reverse().find(item => item.task)?.task || null;
      renderAll();
    }
    saveConversations();
    return result.conversation;
  }

  async function initializeConversationStore() {
    if (conversationStoreReady || conversationStoreLoading) return;
    conversationStoreLoading = true;
    try {
      const migrationKey = STORAGE_KEY + '-server-import-v1';
      if (!localStorage.getItem(migrationKey)) {
        // Retain the original cache as a recoverable migration backup.
        if (!localStorage.getItem(STORAGE_KEY + '-before-server-import')) {
          localStorage.setItem(STORAGE_KEY + '-before-server-import', localStorage.getItem(STORAGE_KEY) || '[]');
        }
        for (const legacy of conversations) await api('/api/conversation/import', legacy);
        localStorage.setItem(migrationKey, 'done');
      }
      const result = await api('/api/conversations');
      conversations = result.conversations.map(item => ({ ...item, messages: [] }));
      if (!conversations.some(item => item.id === activeId)) activeId = conversations[0]?.id || null;
      if (!activeId) {
        const created = await api('/api/conversation/create', { id: uid(), title: '新对话' });
        conversations.unshift(created.conversation);
        activeId = created.conversation.id;
      }
      await loadServerConversation(activeId);
      conversationStoreReady = true;
    } catch (error) {
      composerStatus.textContent = '会话同步失败，旧记录仍保留在本机：' + error.message;
    } finally {
      conversationStoreLoading = false;
      refreshActionAvailability();
    }
  }

  async function refreshServerConversations() {
    if (!conversationStoreReady || sending || conversationStoreLoading || speechState !== 'idle') return;
    conversationStoreLoading = true;
    try {
      const result = await api('/api/conversations');
      conversations = result.conversations.map(item => ({ ...item,
        messages: conversations.find(c => c.id === item.id)?.messages || [] }));
      if (!conversations.some(c => c.id === activeId)) activateConversation(conversations[0]?.id || null);
      if (activeId) await loadServerConversation(activeId);
      else await newConversation();
      renderHistory();
    } catch (_) { /* A temporary outage must not clear cached records. */ }
    finally { conversationStoreLoading = false; }
  }
  window.addEventListener('focus', refreshServerConversations);

  function activateConversation(id) {
    if (id === activeId) return;
    if (activeId) composerDrafts.set(activeId, { text: input.value, reference: pendingReference,
      capability: capabilitySelect.value, pickup: welcomePickupAnnId, dropoff: welcomeReturnAnnId,
      patrolIds: [...patrolSelectedIds], patrolSnapshot: patrolMapSnapshot, patrolRounds: patrolRounds.value });
    pendingReference = null; // The previous draft owns its object URL until restored/deleted.
    clearPendingReference();
    activeId = id;
    activeTask = null;
    mapQueryIds = new Set();
    const draft = composerDrafts.get(id) || {};
    input.value = draft.text || '';
    pendingReference = draft.reference || null;
    if (pendingReference) {
      referencePreviewImage.src = pendingReference.previewUrl;
      referencePreviewName.textContent = pendingReference.name;
      referencePreview.hidden = false;
    }
    capabilitySelect.value = draft.capability || '';
    welcomePickupAnnId = draft.pickup ?? null;
    welcomeReturnAnnId = draft.dropoff ?? null;
    welcomeSelectionRole = null;
    patrolSelectedIds = [...(draft.patrolIds || [])];
    patrolMapSnapshot = draft.patrolSnapshot || null;
    patrolRounds.value = draft.patrolRounds || '-1';
    refreshPatrolSetup();
    refreshWelcomeSetup();
    autoResize();
  }

  function canSwitchConversation() {
    if (speechState === 'idle') return true;
    composerStatus.textContent = '请先结束或取消语音输入，再切换会话';
    return false;
  }

  function ensureConversation() {
    let conversation = currentConversation();
    if (!conversation) {
      conversation = { id: uid(), title: '新对话', createdAt: Date.now(), messages: [] };
      conversations.unshift(conversation);
      activeId = conversation.id;
      saveConversations();
    }
    return conversation;
  }

  async function newConversation() {
    if (!conversationStoreReady || !canSwitchConversation()) return;
    const conversation = { id: uid(), title: '新对话', createdAt: Date.now(), messages: [] };
    try { await api('/api/conversation/create', { id: conversation.id, title: conversation.title }); }
    catch (error) { composerStatus.textContent = error.message; return; }
    conversations.unshift(conversation);
    activateConversation(conversation.id);
    activeTask = null;
    saveConversations();
    renderAll();
    input.focus();
    closeSidebar();
  }

  async function deleteConversation(id) {
    if (!conversationStoreReady || !canSwitchConversation()) return;
    try { await api('/api/conversation/delete', { id }); }
    catch (error) { composerStatus.textContent = error.message; return; }
    conversations = conversations.filter(item => item.id !== id);
    if (activeId === id) activateConversation(conversations[0]?.id || null);
    const draft = composerDrafts.get(id);
    if (draft?.reference?.previewUrl) URL.revokeObjectURL(draft.reference.previewUrl);
    composerDrafts.delete(id);
    if (!activeId) await newConversation();
    else await loadServerConversation(activeId);
    saveConversations();
    renderAll();
  }

  function renderHistory() {
    if (workspaceMode === 'mapping') {
      renderMappingHistory();
      return;
    }
    historyList.replaceChildren();
    for (const conversation of conversations) {
      const row = document.createElement('div');
      row.className = 'history-item' + (conversation.id === activeId ? ' active' : '');
      const select = document.createElement('button');
      select.type = 'button';
      select.className = 'history-select';
      select.textContent = conversation.title || '新对话';
      select.addEventListener('click', async () => {
        if (!conversationStoreReady || !canSwitchConversation()) return;
        activateConversation(conversation.id);
        activeTask = null;
        saveConversations();
        renderAll();
        closeSidebar();
        try { await loadServerConversation(conversation.id); }
        catch (error) { composerStatus.textContent = error.message; }
      });
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'icon-button history-delete';
      remove.setAttribute('aria-label', '删除对话');
      remove.innerHTML = '<svg><use href="#i-trash"/></svg>';
      remove.addEventListener('click', event => {
        event.stopPropagation();
        deleteConversation(conversation.id);
      });
      row.append(select, remove);
      historyList.append(row);
    }
  }

  const mappingStatusLabels = {
    planned: '等待开始', running: '探索中', canceling: '正在停止', canceled: '已停止',
    interrupted: '可继续', uploading: '上传中', ready: '服务器已连接', processing: '检测中',
    server_processing: '生成地图中', complete: '已完成', failed: '失败', captured: '采集完成',
  };

  function mappingStatusText(status) {
    return mappingStatusLabels[status] || status || '未知';
  }

  function renderMappingHistory() {
    historyList.replaceChildren();
    if (!mappingSessions.length) {
      const empty = document.createElement('div');
      empty.className = 'mapping-history-empty';
      empty.textContent = '暂无建图历史';
      historyList.append(empty);
      return;
    }
    for (const session of mappingSessions) {
      const row = document.createElement('div');
      row.className = 'history-item mapping-history-item' + (session.id === selectedMappingId ? ' active' : '');
      const select = document.createElement('button');
      select.type = 'button';
      select.className = 'history-select mapping-history-select';
      const name = document.createElement('span');
      name.textContent = session.name || '未命名建图';
      const meta = document.createElement('small');
      meta.textContent = `${mappingStatusText(session.status)} · ${session.frames || 0}/${session.estimated_frames || 0} 帧 · ${session.objects || 0} 物体`;
      select.append(name, meta);
      select.addEventListener('click', () => {
        selectedMappingId = session.id;
        loadMappingSession(session.id);
        renderHistory();
        closeSidebar();
      });
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'icon-button history-delete';
      remove.setAttribute('aria-label', '删除建图历史');
      remove.innerHTML = '<svg><use href="#i-trash"/></svg>';
      remove.disabled = ['running', 'canceling', 'uploading', 'server_processing', 'processing'].includes(session.status);
      remove.addEventListener('click', async event => {
        event.stopPropagation();
        if (!confirm(`确定删除建图记录“${session.name || '未命名建图'}”吗？`)) return;
        try {
          const response = await fetch(`/api/mapping/session?session_id=${encodeURIComponent(session.id)}`, { method: 'DELETE' });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || `删除失败 (${response.status})`);
          mappingSessions = data.sessions || [];
          if (selectedMappingId === session.id) {
            selectedMappingId = data.active_id || mappingSessions[0]?.id || null;
            selectedMapping = null;
          }
          renderHistory();
          if (selectedMappingId) await loadMappingSession(selectedMappingId);
          else renderMappingWorkspace();
        } catch (error) {
          mappingHeaderStatus.textContent = error.message;
        }
      });
      row.append(select, remove);
      historyList.append(row);
    }
  }

  async function switchWorkspace(mode = null) {
    workspaceMode = mode || (workspaceMode === 'mapping' ? 'function' : 'mapping');
    localStorage.setItem('jaka-vision-workspace', workspaceMode);
    app.classList.toggle('is-mapping', workspaceMode === 'mapping');
    sidebarHeading.textContent = workspaceMode === 'mapping' ? '建图历史' : '对话历史';
    workspaceSwitch.querySelector('span').textContent = workspaceMode === 'mapping' ? '功能' : '建图';
    workspaceSwitch.querySelector('use').setAttribute('href', workspaceMode === 'mapping' ? '#i-camera' : '#i-route');
    workspaceSwitch.setAttribute('aria-label', workspaceMode === 'mapping' ? '切换到功能界面' : '切换到建图界面');
    if (workspaceMode === 'mapping') {
      closeMapPanel();
      await loadMappingSessions();
      renderMappingMap();
    } else {
      renderAll();
    }
    renderHistory();
    closeSidebar();
  }

  async function loadMappingSessions() {
    const result = await api('/api/mapping/sessions');
    mappingSessions = result.sessions || [];
    const known = new Set(mappingSessions.map(item => item.id));
    if (!selectedMappingId || !known.has(selectedMappingId)) {
      selectedMappingId = result.active_id || mappingSessions[0]?.id || null;
    }
    renderHistory();
    if (selectedMappingId) await loadMappingSession(selectedMappingId);
    else {
      selectedMapping = null;
      renderMappingWorkspace();
    }
  }

  async function loadMappingSession(sessionId) {
    const result = await api(`/api/mapping/status?session_id=${encodeURIComponent(sessionId)}`);
    selectedMapping = result.session || null;
    selectedMappingId = selectedMapping?.id || sessionId;
    renderMappingWorkspace();
  }

  function mappingPreviewUrl(name) {
    if (!selectedMapping?.id || !name) return '';
    return `/api/mapping/preview?session_id=${encodeURIComponent(selectedMapping.id)}&name=${encodeURIComponent(name)}`;
  }

  function renderMappingMetrics(session) {
    const values = [
      [session?.progress?.frames || 0, '已采集'],
      [session?.bridge?.acked || 0, '已上传'],
      [session?.bridge?.processed || 0, '已处理'],
      [session?.result?.objects || session?.bridge?.objects || 0, '物体'],
    ];
    mappingMetrics.replaceChildren();
    for (const [value, label] of values) {
      const item = document.createElement('span');
      const number = document.createElement('b');
      number.textContent = String(value);
      item.append(number, document.createTextNode(label));
      mappingMetrics.append(item);
    }
  }

  function renderMappingPreviews(session) {
    const previews = session?.previews || [];
    mappingPreviewList.replaceChildren();
    mappingPreviewCount.textContent = `${previews.length} 帧`;
    const latest = selectedPreviewName && previews.includes(selectedPreviewName)
      ? selectedPreviewName : (session?.latest_preview || previews.at(-1));
    selectedPreviewName = latest || null;
    if (latest) {
      const url = mappingPreviewUrl(latest);
      mappingLatestPreview.src = url;
      mappingLatestPreview.hidden = false;
      mappingPreviewEmpty.hidden = true;
      mappingLatestPreviewButton.disabled = false;
    } else {
      mappingLatestPreview.removeAttribute('src');
      mappingLatestPreview.hidden = true;
      mappingPreviewEmpty.hidden = false;
      mappingLatestPreviewButton.disabled = true;
    }
    for (const name of previews.slice().reverse()) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'mapping-preview-thumb' + (name === latest ? ' selected' : '');
      const image = document.createElement('img');
      image.src = mappingPreviewUrl(name);
      image.alt = `检测预览 ${name}`;
      image.loading = 'lazy';
      button.append(image);
      button.addEventListener('click', () => {
        selectedPreviewName = name;
        renderMappingPreviews(selectedMapping);
      });
      mappingPreviewList.append(button);
    }
  }

  function renderMappingWorkspace() {
    const session = selectedMapping;
    const status = session?.status || 'idle';
    mappingHeaderStatus.textContent = session ? `${session.name} · ${mappingStatusText(status)}` : '尚未计算路线';
    mappingPreviewStatus.textContent = session ? mappingStatusText(session.bridge?.status || status) : '等待采集';
    const total = Number(session?.plan?.estimated_frames || 0);
    const frames = Number(session?.progress?.frames || 0);
    const pointTotal = Number(session?.plan?.route?.length || 0);
    const completed = Number(session?.progress?.completed?.length || 0);
    const percent = total ? Math.min(100, Math.round(frames / total * 100)) : 0;
    mappingProgressBar.style.width = `${percent}%`;
    mappingProgressText.textContent = session
      ? `${mappingStatusText(status)} · 探索点 ${completed}/${pointTotal} · 采集 ${frames}/${total} 帧`
      : '设置参数后计算路线';
    mappingPlanButton.disabled = ['running', 'canceling', 'uploading', 'processing', 'server_processing'].includes(status);
    mappingStartButton.disabled = status !== 'planned';
    mappingStartButton.hidden = status !== 'planned';
    mappingStopButton.hidden = !['running', 'uploading', 'processing', 'server_processing', 'canceling'].includes(status);
    mappingStopButton.disabled = status === 'canceling';
    mappingResumeButton.hidden = !['canceled', 'interrupted', 'failed'].includes(status);
    mappingResultPanel.hidden = status !== 'complete' || !session?.result?.map_name;
    if (!mappingResultPanel.hidden) {
      const result = session.result;
      mappingResultStats.textContent = `${result.objects} 个物体 · ${result.categories} 类 · ${result.functional_relationships + result.positional_relationships} 条关系 · ${result.invalid_objects} 个无效物体`;
    }
    renderMappingMetrics(session);
    renderMappingPreviews(session);
    renderMappingMap();
  }

  function assistantAvatar() {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<svg><use href="#i-camera"/></svg>';
    return avatar;
  }

  function openImage(url) {
    dialogImage.src = url;
    if (typeof imageDialog.showModal === 'function') imageDialog.showModal();
  }

  function appendInlineMarkdown(parent, value) {
    const tokens = String(value).split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
    for (const token of tokens) {
      if (token.startsWith('**') && token.endsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = token.slice(2, -2);
        parent.append(strong);
      } else if (token.startsWith('`') && token.endsWith('`')) {
        const code = document.createElement('code');
        code.textContent = token.slice(1, -1);
        parent.append(code);
      } else {
        parent.append(document.createTextNode(token));
      }
    }
  }

  function renderRichText(parent, value) {
    parent.classList.add('rich');
    const lines = String(value || '').split(/\r?\n/);
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index += 1; continue; }
      const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
      const numbered = /^\s*\d+[.、]\s*(.+)$/.exec(line);
      if (bullet || numbered) {
        const list = document.createElement(numbered ? 'ol' : 'ul');
        const pattern = numbered ? /^\s*\d+[.、]\s*(.+)$/ : /^\s*[-*]\s+(.+)$/;
        while (index < lines.length) {
          const match = pattern.exec(lines[index]);
          if (!match) break;
          const item = document.createElement('li');
          appendInlineMarkdown(item, match[1]);
          list.append(item);
          index += 1;
        }
        parent.append(list);
        continue;
      }
      const paragraph = document.createElement('p');
      appendInlineMarkdown(paragraph, line.replace(/^#{1,3}\s+/, ''));
      parent.append(paragraph);
      index += 1;
    }
  }

  function taskStatusText(status) {
    if (status === 'interrupted') return '已中断，需重新规划';
    if (status === 'superseded') return '已被新计划替代';
    return ({
      needs_clarification: '需要选择',
      planned: '等待确认', running: '执行中', canceling: '正在停止',
      succeeded: '已完成', anomaly: '发现异常', not_found: '未确认找到', inconclusive: '部分点无法确认', canceled: '已取消', aborted: '已中止', error: '执行失败',
      pending: '等待中', baseline: '已建基线', normal: '无异常', active: '检查中', uncertain: '无法确认', miss: '已排除', match: '已找到'
    })[status] || status || '未知';
  }

  function userFacingError(value) {
    const text = String(value || '').trim();
    if (!text) return '请求未完成，请稍后重试。';
    if (/Internal Server Error|HTTP\s*5\d\d|Connection error|服务失败/i.test(text)) {
      return '视觉服务暂时不可用，机器人没有继续执行，请稍后重试。';
    }
    if (/UNKNOWN_ERROR|accessible_point_query|没有可到达/i.test(text)) {
      return '目标附近暂时没有可到达的位置，机器人未继续移动。';
    }
    if (/对话路由|未知对话模式|mode=/.test(text)) {
      return '我没有理解这条指令，请说明要导航、观察、巡逻，还是寻找物品。';
    }
    if (/find_object vision comparison failed|寻物识别结果/.test(text)) {
      return '暂时无法确认现场是否发现目标物品，请稍后重试。';
    }
    return text;
  }

  function stepTypeText(type) {
    return ({ navigate: '导航', cruise: '多点移动', patrol: '视觉巡逻', observe: '观察', find_object: '前往并寻找', wait_guest: '等待并确认客人', return: '返回本次出发点', cancel: '取消移动' })[type] || type;
  }

  function renderFindRoutePreview(parent, task) {
    const route = (task.route_points || []).filter(point => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y)));
    if (!route.length || !mapData?.objects?.length) return;
    const width = 720;
    const height = 960;
    const margin = 42;
    const context = mapData.objects.map(object => {
      const position = object.nav_xy || object.floor_xy;
      return { object, x: Number(position?.[0]), y: Number(position?.[1]) };
    }).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
    const rawWorldPoints = context.length
      ? context.map(point => [point.x, point.y])
      : route.map(point => [Number(point.x), Number(point.y)]);
    const meanX = rawWorldPoints.reduce((sum, point) => sum + point[0], 0) / rawWorldPoints.length;
    const meanY = rawWorldPoints.reduce((sum, point) => sum + point[1], 0) / rawWorldPoints.length;
    const covariance = rawWorldPoints.reduce((value, point) => {
      const dx = point[0] - meanX;
      const dy = point[1] - meanY;
      value.xx += dx * dx;
      value.xy += dx * dy;
      value.yy += dy * dy;
      return value;
    }, { xx: 0, xy: 0, yy: 0 });
    const principalAngle = .5 * Math.atan2(2 * covariance.xy, covariance.xx - covariance.yy);
    // 拟物体地图是走廊型长图，将地图长轴稳定地放在竖直方向。
    const verticalRotation = Math.PI / 2 - principalAngle;
    const cosRoute = Math.cos(verticalRotation);
    const sinRoute = Math.sin(verticalRotation);
    const orient = (x, y) => {
      const dx = Number(x) - meanX;
      const dy = Number(y) - meanY;
      return [meanX + dx * cosRoute - dy * sinRoute, meanY + dx * sinRoute + dy * cosRoute];
    };
    const worldPoints = rawWorldPoints.map(point => orient(point[0], point[1]));
    let minX = Math.min(...worldPoints.map(point => point[0]));
    let maxX = Math.max(...worldPoints.map(point => point[0]));
    let minY = Math.min(...worldPoints.map(point => point[1]));
    let maxY = Math.max(...worldPoints.map(point => point[1]));
    let rangeX = Math.max(2, maxX - minX);
    let rangeY = Math.max(2, maxY - minY);
    const plotRatio = (width - margin * 2) / (height - margin * 2);
    if (rangeX / rangeY < plotRatio) {
      const next = rangeY * plotRatio;
      minX -= (next - rangeX) / 2;
      rangeX = next;
    } else {
      const next = rangeX / plotRatio;
      minY -= (next - rangeY) / 2;
      rangeY = next;
    }
    const project = (x, y) => {
      const point = orient(x, y);
      return [
        margin + (point[0] - minX) / rangeX * (width - margin * 2),
        height - margin - (point[1] - minY) / rangeY * (height - margin * 2),
      ];
    };
    const points = route.map(point => ({ ...point, pixel: project(point.x, point.y) }));
    let robotPixel = null;
    let robotTheta = 0;
    if (robotData?.pose) {
      const nearest = Math.min(...context.map(point => Math.hypot(point.x - Number(robotData.pose.x), point.y - Number(robotData.pose.y))));
      if (nearest < 15) {
        robotPixel = project(robotData.pose.x, robotData.pose.y);
        robotTheta = Number(robotData.pose.theta || 0);
      }
    }
    const figure = document.createElement('figure');
    figure.className = 'find-route-preview';
    const meta = document.createElement('figcaption');
    meta.className = 'find-route-meta';
    const checked = (task.checked_ann_ids || []).length;
    const current = task.current_target_ann_id == null ? '等待执行' : `正在检查 #${task.current_target_ann_id}`;
    const stage = task.current_search_stage || '等待抓拍';
    const snapshots = Number(task.snapshot_count || 0);
    meta.innerHTML = `<strong>寻物路线</strong><span>${route.length} 个候选点</span><span>${checked} 个已检查</span><span>${current}</span><span>${stage}</span><span>${snapshots} 次抓拍</span>`;
    const svg = svgNode('svg', {
      class: 'find-route-map',
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': '拟物体地图候选点和寻物路线',
    });
    const routeIds = new Set(route.map(point => String(point.ann_id)));
    for (const value of context) {
      if (routeIds.has(String(value.object.ann_id))) continue;
      const [x, y] = project(value.x, value.y);
      const group = svgNode('g', { class: 'find-context-object' });
      group.append(svgNode('circle', { cx: x, cy: y, r: 8, class: 'find-route-halo' }));
      group.append(svgNode('image', {
        href: `/map-icons/${objectSymbolKind(value.object)}.svg`, x: x - 5, y: y - 5,
        width: 10, height: 10, preserveAspectRatio: 'xMidYMid meet', class: 'find-route-icon',
      }));
      const title = svgNode('title');
      title.textContent = value.object.category_zh || value.object.category || '地图物体';
      group.append(title);
      svg.append(group);
    }
    const remaining = points.filter(point => {
      const value = task.candidate_status?.[String(point.ann_id)] || point.verdict || 'candidate';
      return value === 'candidate' || value === 'active';
    });
    const routePixels = (robotPixel ? [robotPixel] : []).concat(remaining.map(point => point.pixel));
    if (routePixels.length > 1) {
      let pathData = `M ${routePixels[0][0]} ${routePixels[0][1]}`;
      for (let index = 1; index < routePixels.length; index += 1) {
        const point = routePixels[index];
        if (index < routePixels.length - 1) {
          const next = routePixels[index + 1];
          pathData += ` Q ${point[0]} ${point[1]} ${(point[0] + next[0]) / 2} ${(point[1] + next[1]) / 2}`;
        } else pathData += ` L ${point[0]} ${point[1]}`;
      }
      svg.append(svgNode('path', { d: pathData, class: 'find-route-line' }));
    }
    for (const point of points) {
      const value = task.candidate_status?.[String(point.ann_id)] || point.verdict || 'candidate';
      const group = svgNode('g', { class: `find-route-point ${value}` });
      const object = mapData.objects.find(item => String(item.ann_id) === String(point.ann_id));
      const x = point.pixel[0];
      const y = point.pixel[1];
      group.append(svgNode('circle', { cx: x, cy: y, r: 15, class: 'find-route-halo' }));
      group.append(svgNode('image', {
        href: `/map-icons/${objectSymbolKind(object || {})}.svg`, x: x - 9, y: y - 9,
        width: 18, height: 18, preserveAspectRatio: 'xMidYMid meet', class: 'find-route-icon',
      }));
      if (value === 'match' || value === 'miss' || value === 'uncertain') {
        const verdict = svgNode('text', { x: x + 13, y: y - 12, class: 'find-route-verdict' });
        verdict.textContent = value === 'match' ? '✓' : value === 'uncertain' ? '?' : '×';
        group.append(verdict);
      }
      const title = svgNode('title');
      title.textContent = `${object?.category_zh || object?.category || '候选点'} #${point.ann_id}`;
      group.append(title);
      svg.append(group);
    }
    if (robotPixel) {
      const theta = robotTheta;
      const size = 15;
      const arrow = [[size, 0], [-size * .65, size * .62], [-size * .35, 0], [-size * .65, -size * .62]]
        .map(([dx, dy]) => `${robotPixel[0] + Math.cos(theta) * dx + Math.sin(theta) * dy},${robotPixel[1] + Math.sin(theta) * dx - Math.cos(theta) * dy}`).join(' ');
      svg.append(svgNode('polygon', { points: arrow, class: 'find-route-robot' }));
    }
    figure.append(meta, svg);
    parent.append(figure);
  }

  function executionBlocker(robot) {
    if (!robot?.online) return '底盘离线或状态未知，不能启动任务';
    if (robot.estop_state === true) return '底盘急停中：请现场确认安全并检查急停装置，系统不会自动解除';
    if (robot.estop_state !== false) return '急停状态未知，不能启动任务';
    return '';
  }

  function renderTaskTool(parent, message) {
    const task = message.task;
    if (!task) return;
    const tool = document.createElement('section');
    tool.className = `task-tool${task.kind === 'find_object' ? ' find-object-task' : ''}`;
    const header = document.createElement('div');
    header.className = 'task-tool-header';
    header.innerHTML = '<svg><use href="#i-route"/></svg>';
    const title = document.createElement('div');
    title.className = 'task-tool-title';
    title.textContent = task.status === 'needs_clarification'
      ? '选择具体目标'
      : (task.kind === 'find_object'
        ? '拟物体地图寻物'
        : (task.kind === 'welcome'
          ? '迎宾接待'
          : (task.kind === 'patrol' ? '多点视觉巡逻' : '任务计划')));
    const status = document.createElement('span');
    status.className = 'task-status ' + (task.status || '');
    if (task.skill?.name && task.status !== 'needs_clarification') title.textContent = task.skill.name;
    status.textContent = taskStatusText(task.status);
    header.append(title, status);
    tool.append(header);

    if (task.kind !== 'find_object') {
      const understanding = document.createElement('div');
      understanding.className = 'task-understanding';
      understanding.textContent = task.understanding || task.instruction || '';
      tool.append(understanding);
    }

    if (task.kind === 'welcome' && (task.current_stage || task.result_text)) {
      const welcomeState = document.createElement('div');
      welcomeState.className = 'semantic-reasoning';
      welcomeState.textContent = task.result_text || task.current_stage;
      tool.append(welcomeState);
      const pending = task.guest_confirmation;
      if (task.status === 'running' && !task.cancel_requested && pending?.status === 'pending') {
        const panel = document.createElement('div');
        panel.className = 'task-understanding';
        const photo = document.createElement('img');
        photo.src = pending.image_url;
        photo.alt = '等待主人确认的现场客人照片';
        photo.style.cssText = 'display:block;width:100%;max-width:560px;border-radius:8px';
        photo.addEventListener('click', () => openImage(pending.image_url));
        const note = document.createElement('p');
        note.textContent = `请确认照片中是否为目标客人。确认后机器人将前往${task.return_name || '送客点'}；未确认不会返程。`;
        const actions = document.createElement('div');
        actions.className = 'task-actions';
        const reject = document.createElement('button');
        reject.type = 'button';
        reject.className = 'command-button';
        reject.textContent = '不是目标，继续等待';
        const accept = document.createElement('button');
        accept.type = 'button';
        accept.className = 'command-button primary';
        accept.textContent = '确认客人，前往送客点';
        accept.disabled = true;
        reject.disabled = !navigator.onLine;
        // Never offer a motion confirmation before its evidence has loaded.
        photo.addEventListener('load', () => { accept.disabled = !navigator.onLine || Boolean(executionBlocker(robotData)); });
        photo.addEventListener('error', () => { accept.disabled = true; note.textContent = '现场照片加载失败，请恢复连接或刷新后再确认。'; });
        const decide = async accepted => {
          accept.disabled = reject.disabled = true;
          await confirmWelcomeGuest(task, pending.id, accepted);
        };
        accept.addEventListener('click', () => decide(true));
        reject.addEventListener('click', () => decide(false));
        actions.append(reject, accept);
        panel.append(photo, note, actions);
        tool.append(panel);
      }
    }
    if (task.kind === 'find_object') {
      renderFindRoutePreview(tool, task);
      if (task.search_attempts?.length) {
        const attempts = document.createElement('div');
        attempts.className = 'find-attempts';
        for (const value of task.search_attempts.slice(-12)) {
          const figure = document.createElement('figure');
          figure.className = `find-attempt ${value.verdict === 'match' ? 'match' : ''}`;
          const image = document.createElement('img');
          image.src = value.image_url;
          image.alt = value.stage || '寻物搜索画面';
          image.addEventListener('click', () => openImage(value.image_url));
          const caption = document.createElement('figcaption');
          caption.textContent = `${value.verdict === 'match' ? '✓' : value.verdict === 'uncertain' ? '? 识别无效，未确认' : '×'} ${value.stage || '搜索视角'}`;
          figure.append(image, caption);
          attempts.append(figure);
        }
        tool.append(attempts);
      }
      if (task.result_text) {
        const result = document.createElement('div');
        result.className = 'task-understanding';
        result.textContent = task.result_text;
        tool.append(result);
      }
    }

    if (task.kind === 'patrol') {
      const reasoning = document.createElement('div');
      reasoning.className = 'semantic-reasoning';
      const pointCount = Object.keys(task.patrol_status_by_ann || {}).length;
      const round = Number(task.patrol_round || 0);
      reasoning.textContent = `拟物体地图提供 ${pointCount} 个巡逻锚点和访问顺序；第 1 轮建立各点视觉基线，之后每轮在同一点比较前后画面。当前第 ${round} 轮。`;
      tool.append(reasoning);
      const pointList = document.createElement('div');
      pointList.className = 'candidate-status-list';
      for (const step of task.steps || []) {
        if (step.type !== 'patrol') continue;
        for (let index = 0; index < (step.target_ann_ids || []).length; index += 1) {
          const annId = step.target_ann_ids[index];
          const value = task.patrol_status_by_ann?.[String(annId)] || 'pending';
          const chip = document.createElement('span');
          chip.className = `candidate-status-item ${value === 'anomaly' ? 'miss' : (value === 'normal' ? 'match' : value)}`;
          const symbol = value === 'anomaly' ? '!' : (value === 'normal' ? '✓' : (value === 'active' ? '●' : '○'));
          chip.textContent = `${symbol} ${(step.target_names || [])[index] || '巡逻点'} #${annId} · ${taskStatusText(value)}`;
          pointList.append(chip);
        }
      }
      tool.append(pointList);
      if (task.result_text) {
        const result = document.createElement('div');
        result.className = 'task-understanding';
        result.textContent = task.result_text;
        tool.append(result);
      }
    }

    if (task.status === 'needs_clarification' && task.clarification) {
      const question = document.createElement('div');
      question.className = 'clarification-question';
      question.textContent = task.clarification.question || '请选择具体目标';
      const options = document.createElement('div');
      options.className = 'clarification-options';
      for (const candidate of task.candidates || []) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'command-button clarification-option';
        const id = document.createElement('span');
        id.className = 'clarification-option-id';
        id.textContent = `#${candidate.ann_id}`;
        const name = document.createElement('span');
        name.textContent = candidate.name || '目标';
        const position = document.createElement('span');
        position.className = 'clarification-option-position';
        position.textContent = `(${Number(candidate.x).toFixed(2)}, ${Number(candidate.y).toFixed(2)})`;
        button.append(id, name, position);
        button.addEventListener('click', () => {
          selectedObjectId = candidate.ann_id;
          mapQueryIds = new Set([String(candidate.ann_id)]);
          renderMap();
          renderObjectInspector();
          sendMessage(`${task.instruction}（目标已选定：${candidate.name}，ann_id=${candidate.ann_id}）`);
        });
        options.append(button);
      }
      tool.append(question, options);
    }

    if (task.kind !== 'find_object' && task.steps?.length) {
      const steps = document.createElement('ol');
      steps.className = 'task-steps';
      for (const step of task.steps || []) {
      const row = document.createElement('li');
      row.className = 'task-step ' + (step.status || 'pending');
      const index = document.createElement('span');
      index.className = 'step-index';
      index.textContent = String((step.index ?? 0) + 1);
      const main = document.createElement('div');
      main.className = 'step-main';
      const name = document.createElement('div');
      name.className = 'step-name';
      name.textContent = stepTypeText(step.type);
      const detail = document.createElement('div');
      detail.className = 'step-detail';
      const targets = (step.target_names || []).join('、');
      detail.textContent = [targets, step.reason || step.question || ''].filter(Boolean).join(' · ');
      main.append(name, detail);
      const state = document.createElement('span');
      state.className = 'step-state';
      state.textContent = taskStatusText(step.status);
      row.append(index, main, state);
        steps.append(row);
      }
      tool.append(steps);
    }

    if (task.observations?.length) {
      const observations = document.createElement('div');
      observations.className = 'task-observations';
      for (const value of task.observations) {
        if (task.kind === 'welcome' && task.status === 'running' && !task.cancel_requested
            && task.guest_confirmation?.status === 'pending'
            && value.image_url === task.guest_confirmation.image_url) continue;
        const block = document.createElement('div');
        block.className = 'task-observation';
        if (value.image_url) {
          const image = document.createElement('img');
          image.src = value.image_url;
          image.alt = '到达目标后的现场画面';
          image.addEventListener('click', () => openImage(value.image_url));
          block.append(image);
        }
        const answer = document.createElement('div');
        answer.className = 'message-text rich';
        const confidence = value.confidence ? `（置信度：${value.confidence}）` : '';
        renderRichText(answer, (value.text || '观察完成') + confidence);
        block.append(answer);
        observations.append(block);
      }
      tool.append(observations);
    }
    if (task.kind === 'patrol' && Object.keys(task.patrol_baselines || {}).length) {
      const baselines = document.createElement('div');
      baselines.className = 'patrol-baselines';
      for (const value of Object.values(task.patrol_baselines || {})) {
        const figure = document.createElement('figure');
        figure.className = 'patrol-baseline-card';
        const image = document.createElement('img');
        image.src = value.image_url;
        image.alt = `${value.name || '巡逻点'}基线画面`;
        image.addEventListener('click', () => openImage(value.image_url));
        const caption = document.createElement('figcaption');
        caption.textContent = `${value.name || '巡逻点'} #${value.ann_id} · 当前基线`;
        figure.append(image, caption);
        baselines.append(figure);
      }
      tool.append(baselines);
    }
    if (task.kind === 'patrol' && task.patrol_comparisons?.length) {
      const comparisons = document.createElement('div');
      comparisons.className = 'patrol-comparisons';
      for (const value of [...task.patrol_comparisons].reverse()) {
        const block = document.createElement('article');
        block.className = `patrol-comparison ${value.abnormal ? 'anomaly' : ''}`;
        const comparisonHeader = document.createElement('div');
        comparisonHeader.className = 'patrol-comparison-header';
        const comparisonTitle = document.createElement('strong');
        comparisonTitle.textContent = `${value.name || '巡逻点'} #${value.ann_id} · 第 ${value.round} 轮`;
        const verdict = document.createElement('span');
        verdict.className = 'patrol-verdict';
        verdict.textContent = value.abnormal ? '发现异常' : '未见异常';
        comparisonHeader.append(comparisonTitle, verdict);
        const pair = document.createElement('div');
        pair.className = 'patrol-image-pair';
        for (const [url, label] of [[value.before_image_url, '上次基线'], [value.after_image_url, '本次画面']]) {
          const figure = document.createElement('figure');
          const image = document.createElement('img');
          image.src = url;
          image.alt = label;
          image.addEventListener('click', () => openImage(url));
          const caption = document.createElement('figcaption');
          caption.textContent = label;
          figure.append(image, caption);
          pair.append(figure);
        }
        const summary = document.createElement('div');
        summary.className = 'message-text rich';
        summary.textContent = `${value.summary || '比较完成'}（置信度：${value.confidence || 'low'}）`;
        block.append(comparisonHeader, pair, summary);
        if (value.changes?.length) {
          const list = document.createElement('ul');
          list.className = 'patrol-change-list';
          for (const change of value.changes) {
            const item = document.createElement('li');
            item.textContent = `${change.item || '物体'}：${change.detail || change.type || '发生变化'}`;
            list.append(item);
          }
          block.append(list);
        }
        comparisons.append(block);
      }
      tool.append(comparisons);
    }
    if (task.error) {
      const error = document.createElement('div');
      error.className = 'task-error';
      error.textContent = userFacingError(task.error);
      tool.append(error);
    }

    if (task.status === 'planned' || task.status === 'running' || task.status === 'canceling') {
      const actions = document.createElement('div');
      actions.className = 'task-actions';
      if (task.status === 'planned') {
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'command-button';
        cancel.textContent = '取消';
        cancel.addEventListener('click', () => cancelTask(task.id));
        const execute = document.createElement('button');
        execute.type = 'button';
        execute.className = 'command-button primary';
        execute.textContent = '开始执行';
        const blocker = executionBlocker(robotData);
        execute.disabled = Boolean(blocker) || !navigator.onLine;
        if (blocker) {
          const warning = document.createElement('div');
          warning.className = 'task-error';
          warning.textContent = blocker;
          tool.append(warning);
        }
        execute.addEventListener('click', () => executeTask(task.id));
        actions.append(cancel, execute);
      } else {
        const stop = document.createElement('button');
        stop.type = 'button';
        stop.className = 'command-button';
        stop.textContent = task.status === 'canceling' ? '正在停止' : '停止任务';
        stop.disabled = task.status === 'canceling';
        stop.addEventListener('click', () => cancelTask(task.id));
        actions.append(stop);
      }
      tool.append(actions);
    }
    parent.append(tool);
  }

  function renderMessages() {
    const keepAtBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
    const conversation = ensureConversation();
    messageList.replaceChildren();
    messageList.classList.toggle('wide', conversation.messages.some(message => message.task?.kind === 'find_object'));
    emptyState.classList.toggle('hidden', conversation.messages.length > 0);
    titleNode.textContent = conversation.title || '新对话';

    for (const message of conversation.messages) {
      const article = document.createElement('article');
      article.className = 'message ' + message.role;
      if (message.task?.kind === 'find_object') article.classList.add('find-task');
      if (message.role === 'user') {
        const body = document.createElement('div');
        body.className = 'message-body';
        const text = document.createElement('div');
        text.textContent = message.text;
        body.append(text);
        if (message.referenceImage) {
          const image = document.createElement('img');
          image.className = 'reference-message-image';
          image.src = message.referenceImage;
          image.alt = '用户上传的寻物参考图';
          image.addEventListener('click', () => openImage(message.referenceImage));
          body.append(image);
        }
        article.append(body);
      } else {
        article.append(assistantAvatar());
        const content = document.createElement('div');
        content.className = 'message-content';
        if (message.image) {
          const figure = document.createElement('figure');
          figure.className = 'capture-figure';
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'capture-button';
          button.setAttribute('aria-label', '查看拍摄大图');
          const image = document.createElement('img');
          image.src = message.image;
          image.alt = '机器人拍摄的现场画面';
          button.append(image);
          button.addEventListener('click', () => openImage(message.image));
          const meta = document.createElement('figcaption');
          meta.className = 'capture-meta';
          meta.textContent = message.imageSource || '现场拍摄';
          figure.append(button, meta);
          content.append(figure);
        }
        const text = document.createElement('div');
        text.className = 'message-text';
        if (message.state === 'routing' || message.state === 'planning' || message.state === 'capturing' || message.state === 'thinking') {
          text.classList.add('pending');
          const line = document.createElement('span');
          line.className = 'loading-line';
          const dot = document.createElement('span');
          dot.className = 'loading-dot';
          const label = document.createElement('span');
          label.textContent = message.agentProgress || (message.state === 'routing'
            ? '正在判断是否需要查看现场'
            : (message.state === 'planning'
              ? '小卡 正在结合拟物体地图规划任务'
              : (message.state === 'capturing' ? '正在拍摄现场画面' : '正在分析画面')));
          line.append(dot, label);
          text.append(line);
        } else {
          if (message.state === 'error') {
            text.textContent = message.text || '';
            text.classList.add('error');
          } else {
            renderRichText(text, message.text || '');
          }
        }
        if (!(message.task?.kind === 'find_object' && message.state === 'done')) content.append(text);
        if (message.targetChoices?.length) {
          const options = document.createElement('div');
          options.className = 'task-actions';
          for (const candidate of message.targetChoices) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'command-button';
            button.textContent = `${candidate.category_zh || candidate.category} #${candidate.ann_id} · ${candidate.position || '位置描述缺失'}`;
            button.addEventListener('click', () => sendMessage(`选择 #${candidate.ann_id}，继续刚才的任务`, {
              ann_id: candidate.ann_id, snapshot: message.targetChoiceSnapshot
            }));
            options.append(button);
          }
          content.append(options);
        }
        if (message.execution && !message.task && (message.execution.scope || message.responseKind === 'task_status') && (message.toolTrace?.length || message.incomplete || message.responseKind === 'task_status')) {
          const execution = document.createElement('div');
          execution.className = 'capture-meta';
          execution.textContent = `${message.execution.scope === 'current_turn' ? '本轮执行状态' : '关联任务状态'}：${message.execution.label}`;
          content.append(execution);
        }
        if (message.toolTrace?.length) {
          const details = document.createElement('details');
          const summary = document.createElement('summary');
          summary.textContent = `本轮使用了 ${message.toolTrace.length} 次工具`;
          details.append(summary);
          const names = { get_skill_context: '核对任务资料和默认点位', observe_scene: '查看现场', inspect_previous_scene: '查看历史照片', inspect_reference: '查看上传图片', query_map: '查询地图', compare_map_positions: '核对空间关系', list_map_categories: '查看地图类别', ask_user: '澄清需求', query_project_info: '查询项目资料', get_robot_status: '查询状态', propose_task: '生成待确认计划', plan_patrol: '视觉巡逻技能', plan_welcome: '迎宾接待技能', plan_find_object: '参考图寻物技能', plan_inspect_location: '到点查看技能' };
          names.recall_conversation = '查阅本会话记忆';
          names.inspect_memory_image = '分析历史任务照片';
          names.plan_navigate = '地点导航规划';
          for (const step of message.toolTrace) {
            const row = document.createElement('div');
            row.textContent = `${names[step.tool] || '工具'}：${step.ok ? '完成' : '未成功'}`;
            details.append(row);
          }
          content.append(details);
        }
        if (message.task) renderTaskTool(content, message);
        article.append(content);
      }
      messageList.append(article);
    }
    if (keepAtBottom) requestAnimationFrame(() => { viewport.scrollTop = viewport.scrollHeight; });
  }

  function renderAll() {
    renderHistory();
    renderMessages();
  }

  function svgNode(name, attributes = {}) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  }

  function renderMappingMap() {
    mappingMapSvg.replaceChildren();
    const plan = selectedMapping?.plan;
    if (!plan?.route?.length || !slamData?.available) {
      mappingMapStage.classList.remove('has-plan');
      return;
    }
    mappingMapStage.classList.add('has-plan');
    const width = Number(slamData.width || 1000);
    const height = Number(slamData.height || 1000);
    const metadata = plan.map || {};
    const resolution = Number(metadata.resolution || slamData.calibration?.resolution || 1);
    const origin = metadata.origin || [
      Number(slamData.calibration?.origin_x || 0),
      Number(slamData.calibration?.origin_y || 0),
      Number(slamData.calibration?.rotation || 0) * Math.PI / 180,
    ];
    const cosYaw = Math.cos(Number(origin[2] || 0));
    const sinYaw = Math.sin(Number(origin[2] || 0));
    const worldToPixel = point => {
      const dx = Number(point[0]) - Number(origin[0]);
      const dy = Number(point[1]) - Number(origin[1]);
      const localX = cosYaw * dx + sinYaw * dy;
      const localY = -sinYaw * dx + cosYaw * dy;
      return [localX / resolution, height - localY / resolution];
    };
    const routePoints = plan.route.map(index => worldToPixel(plan.points[index]));
    const all = routePoints.slice();
    if (robotData?.pose) all.push(worldToPixel([robotData.pose.x, robotData.pose.y]));
    let minX = Math.min(...all.map(point => point[0]));
    let maxX = Math.max(...all.map(point => point[0]));
    let minY = Math.min(...all.map(point => point[1]));
    let maxY = Math.max(...all.map(point => point[1]));
    const padding = Math.max(80, Math.max(maxX - minX, maxY - minY) * .16);
    minX = Math.max(0, minX - padding); maxX = Math.min(width, maxX + padding);
    minY = Math.max(0, minY - padding); maxY = Math.min(height, maxY + padding);
    const stageRatio = Math.max(.8, mappingMapStage.clientWidth / Math.max(1, mappingMapStage.clientHeight));
    let viewWidth = Math.max(200, maxX - minX);
    let viewHeight = Math.max(200, maxY - minY);
    if (viewWidth / viewHeight < stageRatio) {
      const next = viewHeight * stageRatio;
      minX -= (next - viewWidth) / 2; viewWidth = next;
    } else {
      const next = viewWidth / stageRatio;
      minY -= (next - viewHeight) / 2; viewHeight = next;
    }
    minX = Math.max(0, Math.min(width - viewWidth, minX));
    minY = Math.max(0, Math.min(height - viewHeight, minY));
    mappingMapSvg.setAttribute('viewBox', `${minX} ${minY} ${viewWidth} ${viewHeight}`);
    mappingMapSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    mappingMapSvg.append(svgNode('image', {
      href: slamData.image_url, x: 0, y: 0, width, height,
      preserveAspectRatio: 'none', class: 'slam-map-image',
    }));
    mappingMapSvg.append(svgNode('polyline', {
      points: routePoints.map(point => `${point[0]},${point[1]}`).join(' '),
      class: 'mapping-plan-route',
    }));
    const completedOrders = new Set((selectedMapping.progress?.completed || []).map(item => Number(item.order)));
    const failedPoints = new Set((selectedMapping.progress?.failed || []).map(item => Number(item.point)));
    const currentOrder = Number(selectedMapping.progress?.current_order);
    const markerRadius = Math.max(7, Math.min(18, Math.max(viewWidth, viewHeight) * .018));
    routePoints.forEach((point, order) => {
      const pointIndex = Number(plan.route[order]);
      let state = completedOrders.has(order) ? 'complete' : (failedPoints.has(pointIndex) ? 'failed' : 'planned');
      if (['running', 'uploading'].includes(selectedMapping.status) && order === currentOrder && !completedOrders.has(order)) state = 'active';
      const group = svgNode('g', { class: `mapping-point ${state}` });
      group.append(svgNode('circle', { cx: point[0], cy: point[1], r: markerRadius }));
      const label = svgNode('text', { x: point[0], y: point[1] + .5 });
      label.textContent = String(order + 1);
      group.append(label);
      mappingMapSvg.append(group);
    });
    if (robotData?.pose) {
      const [x, y] = worldToPixel([robotData.pose.x, robotData.pose.y]);
      const theta = Number(robotData.pose.theta || 0) - Number(origin[2] || 0);
      const size = markerRadius * 1.65;
      const points = [[size, 0], [-size * .65, size * .62], [-size * .38, 0], [-size * .65, -size * .62]]
        .map(([dx, dy]) => {
          const px = x + Math.cos(theta) * dx + Math.sin(theta) * dy;
          const py = y + Math.sin(theta) * dx - Math.cos(theta) * dy;
          return `${px},${py}`;
        }).join(' ');
      mappingMapSvg.append(svgNode('polygon', { points, class: 'robot-marker' }));
    }
  }

  function categoryColor(category) {
    const value = String(category || '').toLowerCase();
    if (value.includes('接客点') || value.includes('guest pickup')) return '#d97706';
    if (value.includes('送客点') || value.includes('guest drop-off')) return '#2563eb';
    if (value.includes('盆栽') || value.includes('植物') || value.includes('plant')) return '#3f8a5c';
    if (value.includes('灌木') || value.includes('shrub') || value.includes('bush')) return '#4f8b67';
    if (value.includes('电梯') || value.includes('elevator')) return '#2f8f83';
    if (value.includes('门') || value.includes('door')) return '#5579b8';
    if (value.includes('窗') || value.includes('window')) return '#358aa8';
    if (value.includes('海报') || value.includes('poster')) return '#d06b9a';
    if (value.includes('储物柜') || value.includes('cabinet')) return '#4e9a68';
    if (value.includes('椅') || value.includes('chair') || value.includes('armchair')) return '#b65d83';
    if (value.includes('航空箱') || value.includes('箱') || value.includes('case')) return '#3979b7';
    if (value.includes('glass table') || value.includes('coffee table')) return '#b56b9a';
    if (value.includes('study desk') || value.includes('desk')) return '#6b7280';
    if (value.includes('操作台') || value.includes('桌') || value.includes('table') || value.includes('desk')) return '#7465ad';
    if (value.includes('灯') || value.includes('light')) return '#c58a25';
    let hash = 0;
    for (const char of String(category || 'Unknown')) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return categoryPalette[hash % categoryPalette.length];
  }

  function objectSymbolKind(object) {
    const value = `${object.category_zh || ''} ${object.category || ''}`.toLowerCase();
    if (value.includes('接客点') || value.includes('guest pickup')) return 'pickup';
    if (value.includes('送客点') || value.includes('guest drop-off')) return 'dropoff';
    if (value.includes('灌木') || value.includes('shrub') || value.includes('bush')) return 'shrub';
    if (value.includes('盆栽') || value.includes('植物') || value.includes('plant')) return 'plant';
    if (value.includes('电梯门') || value.includes('elevator')) return 'elevator';
    if (value.includes('门') || value.includes('door')) return 'door';
    if (value.includes('窗') || value.includes('window')) return 'window';
    if (value.includes('海报') || value.includes('poster')) return 'poster';
    if (value.includes('储物柜') || value.includes('cabinet')) return 'cabinet';
    if (value.includes('椅') || value.includes('chair') || value.includes('armchair')) return 'chair';
    if (value.includes('航空箱') || value.includes('箱') || value.includes('case')) return 'case';
    if (value.includes('glass table') || value.includes('coffee table')) return 'coffeetable';
    if (value.includes('study desk') || value.includes('desk')) return 'desk';
    if (value.includes('操作台') || value.includes('桌') || value.includes('table') || value.includes('desk')) return 'table';
    if (value.includes('灯') || value.includes('light')) return 'light';
    return 'object';
  }

  function appendObjectIcon(parent, kind, cx, cy, size) {
    const iconSize = size * .78;
    parent.append(svgNode('image', {
      href: `/map-icons/${kind}.svg`,
      x: cx - iconSize / 2,
      y: cy - iconSize / 2,
      width: iconSize,
      height: iconSize,
      preserveAspectRatio: 'xMidYMid meet',
      class: 'object-icon',
    }));
  }

  function relationEndpointId(relation, side) {
    const nested = relation?.[`${side}_obj`] || relation?.[side] || {};
    const direct = relation?.[`${side}_ann_id`] ?? relation?.[`${side}_id`];
    return String(nested?.ann_id ?? nested?.id ?? direct ?? '');
  }

  function sceneRelations() {
    const positional = (mapData?.pos_relationships || []).map(value => ({ ...value, relation_type: 'positional' }));
    const functional = (mapData?.func_relationships || []).map(value => ({ ...value, relation_type: 'functional' }));
    return positional.concat(functional);
  }

  function relationDisplayName(relation) {
    const raw = String(relation?.relation || relation?.type || relation?.name || '').trim();
    const key = raw.toLowerCase().replace(/[\s_-]+/g, ' ');
    const names = {
      near: '邻近', nearby: '邻近', 'next to': '相邻', adjacent: '相邻',
      on: '位于上方', above: '位于上方', below: '位于下方', under: '位于下方',
      left: '位于左侧', right: '位于右侧', behind: '位于后方', 'in front of': '位于前方',
      inside: '位于内部', contains: '包含', supports: '支撑', used_for: '用于',
    };
    if (names[key]) return names[key];
    if (raw && /^[\u4e00-\u9fff]{1,8}$/.test(raw) && !['真除', '淫根', '屶嚇'].includes(raw)) return raw;
    return relation?.relation_type === 'functional' ? '功能关联' : '空间关联';
  }

  function current3DTargetObject() {
    if (!mapData?.objects?.length) return null;
    let id = activeTask?.current_target_ann_id;
    if (id == null && selectedObjectId != null) id = selectedObjectId;
    if (id == null && activeTask?.kind === 'find_object') {
      id = Object.entries(activeTask.candidate_status || {}).find(([, value]) => value === 'active')?.[0];
      if (id == null) id = (activeTask.route_points || []).find(point => {
        const state = activeTask.candidate_status?.[String(point.ann_id)] || 'candidate';
        return state === 'candidate' || state === 'active';
      })?.ann_id;
    }
    if (id == null) {
      const step = (activeTask?.steps || []).find(value => ['running', 'pending'].includes(value.status));
      id = step?.target_ann_id ?? step?.target_ann_ids?.[0];
    }
    return mapData.objects.find(object => String(object.ann_id) === String(id)) || null;
  }

  function sceneInfoColor(value) {
    const colors = {
      '黄色': '#d7a620', '黑色': '#252b2d', '白色': '#edf2f2', '蓝色': '#3c83bd', '绿色': '#449568',
      '红色': '#c6554e', '灰色': '#859095', '银色': '#aeb8bc', '棕色': '#8b6448', '透明': '#b8d8db',
    };
    return colors[String(value || '').trim()] || '#78909c';
  }

  function render3DInfo() {
    const object = current3DTargetObject();
    scene3dInfo.replaceChildren();
    if (!object) {
      const empty = document.createElement('div');
      empty.className = 'scene-info-empty';
      empty.textContent = '选择物体或开始任务后显示目标信息';
      scene3dInfo.append(empty);
      return;
    }
    const name = object.category_zh || object.category || '目标物体';
    const position = object.nav_xy || object.floor_xy || [];
    const heading = document.createElement('div');
    heading.className = 'scene-info-title';
    const title = document.createElement('strong');
    title.textContent = name;
    const meta = document.createElement('span');
    meta.textContent = `#${object.ann_id}${position.length >= 2 ? ` · (${Number(position[0]).toFixed(2)}, ${Number(position[1]).toFixed(2)})` : ''}`;
    heading.append(title, meta);
    const list = document.createElement('dl');
    list.className = 'scene-info-list';
    const fields = [
      ['颜色', 'color'], ['材质', 'material'], ['尺寸', 'size'], ['状态', 'state'],
      ['位置', 'position'], ['朝向', 'direction'], ['风格', 'style'], ['用途', 'func_desc'],
    ];
    for (const [label, key] of fields) {
      const value = String(object[key] || '').trim();
      if (!value) continue;
      const row = document.createElement('div');
      row.className = 'scene-info-row';
      const term = document.createElement('dt');
      term.textContent = label;
      const description = document.createElement('dd');
      if (key === 'color') {
        const color = document.createElement('span');
        color.className = 'scene-info-color';
        color.style.setProperty('--scene-info-color', sceneInfoColor(value));
        color.innerHTML = '<i aria-hidden="true"></i>';
        color.append(document.createTextNode(value));
        description.append(color);
      } else description.textContent = value;
      row.append(term, description);
      list.append(row);
    }
    const relations = document.createElement('div');
    relations.className = 'scene-info-relations';
    const relationTitle = document.createElement('strong');
    relationTitle.textContent = '物体关系';
    relations.append(relationTitle);
    const objectId = String(object.ann_id);
    let relationCount = 0;
    for (const relation of sceneRelations()) {
      const headId = relationEndpointId(relation, 'head');
      const tailId = relationEndpointId(relation, 'tail');
      if (headId !== objectId && tailId !== objectId) continue;
      const otherId = headId === objectId ? tailId : headId;
      const other = mapData.objects.find(value => String(value.ann_id) === otherId);
      if (!other) continue;
      const item = document.createElement('div');
      item.className = 'scene-info-relation';
      item.textContent = `${relationDisplayName(relation)} · ${other.category_zh || other.category || '物体'} #${other.ann_id}`;
      relations.append(item);
      relationCount += 1;
    }
    if (!relationCount) {
      const item = document.createElement('div');
      item.className = 'scene-info-relation';
      item.textContent = '暂无已知关系';
      relations.append(item);
    }
    scene3dInfo.append(heading, list, relations);
  }

  function sceneBox(object) {
    const geometry = object.geometry || {};
    const center = object.box3d?.center || geometry.center_world || [object.floor_xy?.[0], object.floor_xy?.[1], .5];
    const explicitSize = object.box3d?.size;
    const half = geometry.aabb_half_sizes_world;
    const size = explicitSize || (Array.isArray(half) ? half.map(value => Number(value) * 2) : [.6, .6, 1]);
    if (!Array.isArray(center) || center.length < 2 || !Array.isArray(size) || size.length < 2) return null;
    const values = {
      center: [Number(center[0]), Number(center[1]), Number(center[2] ?? Number(size[2] || 1) / 2)],
      size: [Math.max(.08, Number(size[0]) || .6), Math.max(.08, Number(size[1]) || .6), Math.max(.08, Number(size[2]) || 1)],
    };
    return values.center.every(Number.isFinite) && values.size.every(Number.isFinite) ? values : null;
  }

  function scene3dSignature() {
    return `${mapData?.name || ''}|${(mapData?.objects || []).map(object => {
      const box = sceneBox(object);
      return `${object.ann_id}:${box?.center.join(',')}:${box?.size.join(',')}`;
    }).join('|')}`;
  }

  function makeSceneLabel(text, color) {
    const THREE = window.THREE;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = '600 28px system-ui, sans-serif';
    const width = Math.min(520, Math.max(160, Math.ceil(context.measureText(text).width + 34)));
    canvas.width = width;
    canvas.height = 58;
    context.fillStyle = 'rgba(15,22,23,.88)';
    context.beginPath();
    context.roundRect(1, 1, width - 2, 56, 10);
    context.fill();
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.stroke();
    context.fillStyle = '#f3f7f7';
    context.font = '600 28px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, width / 2, 29, width - 24);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
    sprite.scale.set(width / 260, .28, 1);
    sprite.renderOrder = 8;
    return sprite;
  }

  function rebuild3DRelations() {
    if (!scene3d) return;
    const { THREE, world, items } = scene3d;
    scene3d.relations = [];
    for (const relation of sceneRelations()) {
      const headId = relationEndpointId(relation, 'head');
      const tailId = relationEndpointId(relation, 'tail');
      const head = items.get(headId);
      const tail = items.get(tailId);
      if (!head || !tail || headId === tailId) continue;
      const points = [head.group.position.clone(), tail.group.position.clone()];
      const color = relation.relation_type === 'functional' ? '#d99a35' : '#6f9294';
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: .58 })
      );
      line.renderOrder = 2;
      const label = makeSceneLabel(relationDisplayName(relation), color);
      label.scale.multiplyScalar(.52);
      label.position.copy(points[0]).lerp(points[1], .5);
      label.position.y += .28;
      label.visible = false;
      world.add(line, label);
      scene3d.relations.push({ relation, headId, tailId, line, label });
    }
  }

  function rebuild3DCameraMarker() {
    if (!scene3d) return;
    const { THREE, world } = scene3d;
    const marker = new THREE.Group();
    const optical = new THREE.Group();
    optical.position.y = scene3d.cameraHeight;
    optical.rotation.z = scene3d.cameraPitch;

    const frustumLength = 0.4;
    const halfWidth = Math.tan(scene3d.realViewFovH * Math.PI / 360) * frustumLength;
    const halfHeight = Math.tan(scene3d.realViewFovV * Math.PI / 360) * frustumLength;
    const origin = new THREE.Vector3(0, 0, 0);
    const corners = [
      new THREE.Vector3(frustumLength, halfHeight, -halfWidth),
      new THREE.Vector3(frustumLength, halfHeight, halfWidth),
      new THREE.Vector3(frustumLength, -halfHeight, halfWidth),
      new THREE.Vector3(frustumLength, -halfHeight, -halfWidth),
    ];
    const segments = [];
    for (const corner of corners) segments.push(origin, corner);
    for (let index = 0; index < corners.length; index += 1) {
      segments.push(corners[index], corners[(index + 1) % corners.length]);
    }
    const frustum = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(segments),
      new THREE.LineBasicMaterial({ color: '#25b9c1', transparent: true, opacity: .78 })
    );
    const imagePlane = new THREE.Mesh(
      new THREE.PlaneGeometry(halfWidth * 2, halfHeight * 2),
      new THREE.MeshBasicMaterial({ color: '#25b9c1', transparent: true, opacity: .07, side: THREE.DoubleSide, depthWrite: false })
    );
    imagePlane.rotation.y = Math.PI / 2;
    imagePlane.position.x = frustumLength;
    const opticalCenter = new THREE.Mesh(
      new THREE.SphereGeometry(.09, 18, 12),
      new THREE.MeshLambertMaterial({ color: '#d9f4f2', emissive: '#167d72', emissiveIntensity: .55 })
    );
    optical.add(frustum, imagePlane, opticalCenter);
    marker.add(optical);
    world.add(marker);
    scene3d.cameraMarker = marker;
  }

  function update3DTrackAndCamera() {
    if (!scene3d) return;
    const { THREE, world } = scene3d;
    if (scene3d.trackLine) {
      world.remove(scene3d.trackLine);
      scene3d.trackLine.geometry.dispose();
      scene3d.trackLine.material.dispose();
      scene3d.trackLine = null;
    }
    const track = (robotData?.track || []).filter(point => Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
    if (track.length > 1) {
      const geometry = new THREE.BufferGeometry().setFromPoints(
        track.map(point => new THREE.Vector3(Number(point[0]), .055, -Number(point[1])))
      );
      const material = new THREE.LineBasicMaterial({ color: '#25b9c1', transparent: true, opacity: .92 });
      scene3d.trackLine = new THREE.Line(geometry, material);
      scene3d.trackLine.renderOrder = 5;
      world.add(scene3d.trackLine);
    }
    if (scene3d.cameraMarker && robotData?.pose) {
      scene3d.cameraMarker.visible = true;
      scene3d.cameraMarker.position.set(Number(robotData.pose.x), 0, -Number(robotData.pose.y));
      scene3d.cameraMarker.rotation.y = Number(robotData.pose.theta || 0);
    } else if (scene3d.cameraMarker) scene3d.cameraMarker.visible = false;
  }

  function update3DVisibility() {
    if (!scene3d) return;
    const targetId = String(current3DTargetObject()?.ann_id ?? '');
    const pose = robotData?.pose;
    let visibleCount = 0;
    for (const [id, item] of scene3d.items) {
      let visible = true;
      if (scene3d.realView && pose) {
        const box = sceneBox(item.object);
        const dx = Number(box.center[0]) - Number(pose.x);
        const dy = Number(box.center[1]) - Number(pose.y);
        const distance = Math.hypot(dx, dy);
        let delta = Math.atan2(dy, dx) - Number(pose.theta || 0);
        delta = Math.atan2(Math.sin(delta), Math.cos(delta));
        const verticalAngle = Math.atan2(Number(box.center[2]) - scene3d.cameraHeight, Math.max(distance, .01));
        visible = distance <= scene3d.realViewDistance
          && Math.abs(delta) <= scene3d.realViewFovH * Math.PI / 360
          && Math.abs(verticalAngle - scene3d.cameraPitch) <= scene3d.realViewFovV * Math.PI / 360;
      }
      item.group.visible = visible;
      if (visible) visibleCount += 1;
    }
    for (const value of scene3d.relations || []) {
      const visible = Boolean(scene3d.items.get(value.headId)?.group.visible && scene3d.items.get(value.tailId)?.group.visible);
      value.line.visible = visible;
      value.label.visible = visible && Boolean(targetId) && (value.headId === targetId || value.tailId === targetId);
    }
    scene3dStatus.textContent = scene3d.realView
      ? `真实视野 · ${visibleCount}/${scene3d.items.size} 个物体`
      : `${scene3d.items.size} 个三维物体框`;
    render3DInfo();
  }

  function init3DScene() {
    if (scene3d || !scene3dCanvas) return scene3d;
    if (!window.THREE) {
      scene3dStatus.textContent = '3D 引擎未加载，请检查浏览器网络';
      return null;
    }
    const THREE = window.THREE;
    const renderer = new THREE.WebGLRenderer({ canvas: scene3dCanvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#101617');
    scene.fog = new THREE.Fog('#101617', 16, 42);
    const camera = new THREE.PerspectiveCamera(58, 1, .05, 120);
    scene.add(new THREE.HemisphereLight('#dcecef', '#253234', 1.45));
    const keyLight = new THREE.DirectionalLight('#ffffff', 1.15);
    keyLight.position.set(4, 8, 3);
    scene.add(keyLight);
    const world = new THREE.Group();
    scene.add(world);
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120),
      new THREE.MeshLambertMaterial({ color: '#1a2425', transparent: true, opacity: .88 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -.03;
    scene.add(ground);
    scene3d = {
      THREE, renderer, scene, camera, world, ground, items: new Map(), relations: [], routeLine: null,
      trackLine: null, cameraMarker: null, signature: '', realView: false, realViewDistance: 20,
      realViewFovH: 91, realViewFovV: 65, cameraHeight: 1.464131, cameraPitch: THREE.MathUtils.degToRad(-6.9),
      yawOffset: 0, pitchOffset: 0, fov: 58, followDistance: 2, drag: null, lastPose: null,
    };
    const resize = () => {
      const rect = scene3dStage.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / rect.height;
      camera.updateProjectionMatrix();
    };
    scene3d.resize = resize;
    new ResizeObserver(resize).observe(scene3dStage);
    scene3dStage.addEventListener('pointerdown', event => {
      scene3d.drag = { x: event.clientX, y: event.clientY, yaw: scene3d.yawOffset, pitch: scene3d.pitchOffset };
      scene3dStage.classList.add('dragging');
      scene3dStage.setPointerCapture(event.pointerId);
    });
    scene3dStage.addEventListener('pointermove', event => {
      if (!scene3d.drag) return;
      scene3d.yawOffset = scene3d.drag.yaw - (event.clientX - scene3d.drag.x) * .006;
      scene3d.pitchOffset = Math.max(-.6, Math.min(.6, scene3d.drag.pitch - (event.clientY - scene3d.drag.y) * .004));
      update3DCamera();
    });
    const stopDrag = () => { scene3d.drag = null; scene3dStage.classList.remove('dragging'); };
    scene3dStage.addEventListener('pointerup', stopDrag);
    scene3dStage.addEventListener('pointercancel', stopDrag);
    scene3dStage.addEventListener('wheel', event => {
      event.preventDefault();
      scene3d.fov = Math.max(38, Math.min(82, scene3d.fov + Math.sign(event.deltaY) * 4));
      camera.fov = scene3d.fov;
      camera.updateProjectionMatrix();
    }, { passive: false });
    const animate = time => {
      if (!scene3d) return;
      for (const item of scene3d.items.values()) {
        if (item.state === 'active' || item.state === 'match') {
          const pulse = 1 + Math.sin(time * .004) * .035;
          item.group.scale.setScalar(pulse);
        } else item.group.scale.setScalar(1);
        const distance = item.group.position.distanceTo(camera.position);
        item.label.visible = (distance > 2.5 && distance < 19) || ['active', 'match'].includes(item.state);
      }
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    resize();
    requestAnimationFrame(animate);
    return scene3d;
  }

  function rebuild3DScene() {
    const value = init3DScene();
    if (!value) return;
    const { THREE, world, items } = value;
    while (world.children.length) {
      const child = world.children.pop();
      child.traverse(node => {
        node.geometry?.dispose?.();
        if (Array.isArray(node.material)) node.material.forEach(material => material.dispose?.());
        else node.material?.dispose?.();
      });
    }
    items.clear();
    value.relations = [];
    value.routeLine = null;
    value.trackLine = null;
    value.cameraMarker = null;
    for (const object of mapData?.objects || []) {
      const box = sceneBox(object);
      if (!box) continue;
      const name = object.category_zh || object.category || '物体';
      const color = categoryColor(name);
      const geometry = new THREE.BoxGeometry(box.size[0], box.size[2], box.size[1]);
      const material = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: .38, depthWrite: true });
      const mesh = new THREE.Mesh(geometry, material);
      const edgeMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: .95 });
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edgeMaterial);
      const group = new THREE.Group();
      group.position.set(box.center[0], box.center[2], -box.center[1]);
      group.add(mesh, edges);
      const label = makeSceneLabel(name, color);
      label.position.set(0, box.size[2] / 2 + .34, 0);
      group.add(label);
      world.add(group);
      items.set(String(object.ann_id), { object, group, mesh, edges, label, material, edgeMaterial, color, state: 'normal' });
    }
    rebuild3DRelations();
    rebuild3DCameraMarker();
    reset3DFollowView();
    value.signature = scene3dSignature();
    sync3DHighlights();
    update3DCamera();
  }

  function sync3DHighlights() {
    if (!scene3d) return;
    for (const [id, item] of scene3d.items) {
      const verdict = activeTask?.kind === 'find_object' ? activeTask.candidate_status?.[id] : null;
      const selected = String(selectedObjectId) === id;
      const state = verdict || (selected ? 'active' : 'normal');
      const color = state === 'candidate' ? '#d99a35' : (state === 'active' ? '#25b9c1' : (state === 'miss' ? '#cc5b55' : (state === 'match' ? '#35a879' : item.color)));
      item.state = state;
      item.material.color.set(color);
      item.edgeMaterial.color.set(color);
      item.material.opacity = state === 'miss' ? .12 : (state === 'normal' ? .34 : .72);
      item.edgeMaterial.opacity = state === 'miss' ? .35 : 1;
      item.edges.scale.setScalar(['active', 'match'].includes(state) ? 1.035 : 1);
    }
    if (scene3d.routeLine) {
      scene3d.world.remove(scene3d.routeLine);
      scene3d.routeLine.geometry.dispose();
      scene3d.routeLine.material.dispose();
      scene3d.routeLine = null;
    }
    if (activeTask?.kind === 'find_object' && activeTask.route_points?.length) {
      const path = [];
      if (robotData?.pose) path.push(new scene3d.THREE.Vector3(Number(robotData.pose.x), .035, -Number(robotData.pose.y)));
      for (const point of activeTask.route_points) {
        const verdict = activeTask.candidate_status?.[String(point.ann_id)] || 'candidate';
        if (verdict === 'candidate' || verdict === 'active') {
          path.push(new scene3d.THREE.Vector3(Number(point.x), .035, -Number(point.y)));
        }
      }
      if (path.length > 1) {
        const geometry = new scene3d.THREE.BufferGeometry().setFromPoints(path);
        const material = new scene3d.THREE.LineBasicMaterial({ color: '#d99a35', transparent: true, opacity: .9 });
        scene3d.routeLine = new scene3d.THREE.Line(geometry, material);
        scene3d.routeLine.renderOrder = 4;
        scene3d.world.add(scene3d.routeLine);
      }
    }
    update3DTrackAndCamera();
    update3DVisibility();
  }

  function reset3DFollowView() {
    if (!scene3d) return;
    scene3d.yawOffset = 0;
    scene3d.pitchOffset = 0;
    scene3d.fov = 58;
    scene3d.camera.fov = scene3d.fov;
    scene3d.camera.updateProjectionMatrix();
  }

  function update3DCamera() {
    if (!scene3d) return;
    const { THREE, camera, items } = scene3d;
    let x;
    let y;
    let theta;
    if (robotData?.pose) {
      x = Number(robotData.pose.x);
      y = Number(robotData.pose.y);
      theta = Number(robotData.pose.theta || 0);
    } else {
      const positions = [...items.values()].map(item => item.group.position);
      x = positions.length ? positions.reduce((sum, point) => sum + point.x, 0) / positions.length : 0;
      y = positions.length ? -positions.reduce((sum, point) => sum + point.z, 0) / positions.length : 0;
      theta = 0;
    }
    const baseDirection = new THREE.Vector3(Math.cos(theta), 0, -Math.sin(theta));
    const viewYaw = theta + scene3d.yawOffset;
    const viewPitch = scene3d.cameraPitch + scene3d.pitchOffset;
    const viewDirection = new THREE.Vector3(
      Math.cos(viewPitch) * Math.cos(viewYaw),
      Math.sin(viewPitch),
      -Math.cos(viewPitch) * Math.sin(viewYaw)
    ).normalize();
    camera.position.set(
      x - baseDirection.x * scene3d.followDistance,
      scene3d.cameraHeight,
      -y - baseDirection.z * scene3d.followDistance
    );
    camera.lookAt(camera.position.clone().add(viewDirection.multiplyScalar(6)));
    scene3d.lastPose = `${x.toFixed(3)},${y.toFixed(3)},${theta.toFixed(3)}`;
  }

  function sync3DScene() {
    try {
      const value = init3DScene();
      if (!value || !mapData?.objects?.length) return;
      const signature = scene3dSignature();
      if (signature !== value.signature) rebuild3DScene();
      else {
        sync3DHighlights();
        update3DCamera();
      }
    } catch (error) {
      scene3dStatus.textContent = `3D 场景加载失败：${error.message || error}`;
      console.error('[3D]', error);
    }
  }

  function selectedMapObject() {
    return mapData?.objects?.find(item => String(item.ann_id) === String(selectedObjectId)) || null;
  }

  function mapObjectById(annId) {
    if (annId == null) return null;
    return mapData?.objects?.find(item => String(item.ann_id) === String(annId)) || null;
  }

  function welcomeLocationLabel(annId) {
    const object = mapObjectById(annId);
    if (!object) return '未选择';
    return `${object.category_zh || object.category || '目标'} · #${object.ann_id}`;
  }

  function refreshWelcomeSetup() {
    const enabled = capabilitySelect.value === 'welcome';
    welcomeSetup.hidden = !enabled;
    welcomePickupText.textContent = welcomeLocationLabel(welcomePickupAnnId);
    welcomeReturnText.textContent = welcomeLocationLabel(welcomeReturnAnnId);
    selectWelcomePickupButton.classList.toggle('active', enabled && welcomeSelectionRole === 'pickup');
    selectWelcomeReturnButton.classList.toggle('active', enabled && welcomeSelectionRole === 'return');
    if (!enabled) return;
    if (welcomeSelectionRole === 'pickup') {
      welcomeSelectionHint.textContent = '请在右侧地图点击接人的目的地';
    } else if (welcomeSelectionRole === 'return') {
      welcomeSelectionHint.textContent = '请在右侧地图点击带客人返回的位置';
    } else if (welcomePickupAnnId != null && welcomeReturnAnnId != null) {
      welcomeSelectionHint.textContent = '接人点和返回点已设置，可以发送迎宾指令';
    } else {
      welcomeSelectionHint.textContent = '先选择接人点，再选择带客人返回的位置';
    }
  }

  function beginWelcomeLocationSelection(role) {
    welcomeSelectionRole = role;
    openMapPanel();
    refreshWelcomeSetup();
    composerStatus.textContent = role === 'pickup'
      ? '请在地图中点击接人点'
      : '请在地图中点击返回点';
    renderMap();
  }

  function refreshPatrolSetup() {
    const enabled = capabilitySelect.value === 'patrol';
    patrolSetup.hidden = !enabled;
    if (patrolSelectedIds.length && patrolMapSnapshot !== mapData?.snapshot) {
      patrolSelectedIds = [];
      patrolMapSnapshot = null;
    }
    patrolSelectedList.replaceChildren();
    patrolSelectionCount.textContent = `巡逻选点：${patrolSelectedIds.length} 个`;
    for (const [index, id] of patrolSelectedIds.entries()) {
      const object = mapData?.objects?.find(item => item.ann_id === id);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'command-button';
      button.textContent = `${index + 1}. ${object?.category_zh || object?.category || id} ×`;
      button.addEventListener('click', () => {
        if (sending) return;
        patrolSelectedIds = patrolSelectedIds.filter(value => value !== id);
        refreshPatrolSetup(); renderMap();
      });
      patrolSelectedList.append(button);
    }
    planSelectedPatrol.disabled = sending || speechState !== 'idle' || !conversationStoreReady || !backendOnline || patrolSelectedIds.length < 2;
    patrolSelectionHint.textContent = patrolSelectedIds.length < 2
      ? '至少选择两个地点；不会自动开始移动。' : '仅使用已选地点；生成任务卡后还需确认执行。';
    objectInspector.querySelector('.object-actions').hidden = enabled;
  }

  function togglePatrolPoint(object) {
    if (sending) return;
    refreshPatrolSetup();
    const xy = object.nav_xy || object.floor_xy;
    if (!Number.isInteger(object.ann_id) || (object.lifecycle || 'active') !== 'active' || !Array.isArray(xy)
        || xy.length < 2 || !xy.slice(0, 2).every(Number.isFinite) || !mapData?.snapshot) {
      patrolSelectionHint.textContent = '该图标没有可用导航点，不能加入巡逻。';
      return;
    }
    const index = patrolSelectedIds.indexOf(object.ann_id);
    if (index >= 0) patrolSelectedIds.splice(index, 1);
    else if (patrolSelectedIds.length < 30) patrolSelectedIds.push(object.ann_id);
    else { patrolSelectionHint.textContent = '一次最多选择 30 个地点。'; return; }
    patrolMapSnapshot = mapData.snapshot;
    refreshPatrolSetup();
  }

  function selectedPatrolRequest() {
    if (patrolSelectedIds.length < 2) throw new Error('请至少选择两个巡逻地点');
    if (!patrolMapSnapshot || patrolMapSnapshot !== mapData?.snapshot) throw new Error('地图已变化，请重新选点');
    const ids = [...patrolSelectedIds];
    const rounds = Number(patrolRounds.value);
    const names = ids.map(id => {
      const object = mapData.objects.find(item => item.ann_id === id);
      if (!object) throw new Error('所选点位已失效，请刷新地图重新选择');
      return `${object.category_zh || object.category}（#${id}）`;
    });
    return {target_ann_ids: ids, rounds, map_snapshot: patrolMapSnapshot,
      instruction: `按地图选择顺序巡逻：${names.join(' → ')}；首轮采集基线，${rounds < 0 ? '之后持续检查' : `之后检查 ${rounds} 轮`}，发现异常停止并汇报。`};
  }

  function handleMapObjectSelection(object) {
    selectedObjectId = object.ann_id;
    if (capabilitySelect.value === 'patrol') {
      togglePatrolPoint(object);
      renderMap(); renderObjectInspector();
      return;
    }
    if (capabilitySelect.value === 'welcome' && welcomeSelectionRole) {
      if (welcomeSelectionRole === 'pickup') {
        welcomePickupAnnId = object.ann_id;
        welcomeSelectionRole = 'return';
        composerStatus.textContent = '接人点已选择，请继续点击返回点';
      } else {
        if (String(object.ann_id) === String(welcomePickupAnnId)) {
          composerStatus.textContent = '返回点不能与接人点相同，请选择另一个位置';
          return;
        }
        welcomeReturnAnnId = object.ann_id;
        welcomeSelectionRole = null;
        composerStatus.textContent = '接人点和返回点已设置';
      }
      mapQueryIds = new Set(
        [welcomePickupAnnId, welcomeReturnAnnId]
          .filter(value => value != null)
          .map(String)
      );
      refreshWelcomeSetup();
    }
    renderMap();
    renderObjectInspector();
  }

  function savedTrackColor(track) {
    let hash = 0;
    for (const char of String(track?.id || '')) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return savedTrackPalette[hash % savedTrackPalette.length];
  }

  function storeVisibleTrackIds() {
    localStorage.setItem('jaka-vision-visible-tracks', JSON.stringify([...visibleTrackIds]));
  }

  function renderTrackHistory() {
    trackHistoryList.replaceChildren();
    trackHistoryEmpty.hidden = savedTracks.length > 0;
    for (const track of savedTracks) {
      const id = String(track.id);
      const row = document.createElement('div');
      row.className = 'track-history-row';

      const toggle = document.createElement('input');
      toggle.type = 'checkbox';
      toggle.className = 'track-history-check';
      toggle.checked = visibleTrackIds.has(id);
      toggle.title = '在地图上显示这条轨迹';
      toggle.addEventListener('change', () => {
        if (toggle.checked) visibleTrackIds.add(id);
        else visibleTrackIds.delete(id);
        storeVisibleTrackIds();
        renderMap();
      });

      const swatch = document.createElement('span');
      swatch.className = 'track-history-swatch';
      swatch.style.background = savedTrackColor(track);

      const textBox = document.createElement('div');
      textBox.className = 'track-history-copy';
      const name = document.createElement('div');
      name.className = 'track-history-name';
      name.textContent = track.name || '未命名轨迹';
      name.title = name.textContent;
      const meta = document.createElement('div');
      meta.className = 'track-history-meta';
      meta.textContent = `${Number(track.point_count || track.points?.length || 0)} 点 · ${Number(track.distance_m || 0).toFixed(1)} m`;
      textBox.append(name, meta);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'icon-button track-delete-button';
      remove.title = '删除轨迹';
      remove.setAttribute('aria-label', `删除轨迹 ${name.textContent}`);
      remove.innerHTML = '<svg aria-hidden="true"><use href="#i-trash"></use></svg>';
      remove.addEventListener('click', async () => {
        if (!confirm(`确定删除轨迹“${name.textContent}”吗？`)) return;
        try {
          const result = await api('/api/tracks/delete', { track_id: id });
          savedTracks = result.tracks || [];
          visibleTrackIds.delete(id);
          storeVisibleTrackIds();
          renderTrackHistory();
          renderMap();
        } catch (error) {
          composerStatus.textContent = `轨迹删除失败：${error.message}`;
        }
      });

      row.append(toggle, swatch, textBox, remove);
      trackHistoryList.append(row);
    }
  }

  async function loadTrackHistory() {
    const result = await api('/api/tracks');
    savedTracks = result.tracks || [];
    const knownIds = new Set(savedTracks.map(track => String(track.id)));
    visibleTrackIds = new Set([...visibleTrackIds].filter(id => knownIds.has(id)));
    storeVisibleTrackIds();
    renderTrackHistory();
    renderMap();
  }

  function renderObjectInspector() {
    refreshPatrolSetup();
    const object = selectedMapObject();
    const title = objectInspector.querySelector('.object-inspector-title');
    if (!object) {
      objectInspector.classList.add('empty');
      title.textContent = '选择地图中的物体';
      objectInspectorMeta.textContent = '';
      return;
    }
    objectInspector.classList.remove('empty');
    const name = object.category_zh || object.category || '目标';
    const displayXY = object.nav_xy || object.floor_xy;
    title.textContent = `${name} · #${object.ann_id}`;
    const source = object.nav_xy ? '导航实测坐标' : '拟物体原始坐标';
    objectInspectorMeta.textContent = `${object.func_desc || object.position || ''} · ${source} (${Number(displayXY[0]).toFixed(2)}, ${Number(displayXY[1]).toFixed(2)})`;
  }

  function renderMapLegend() {
    mapLegend.replaceChildren();
    if (!mapData) return;
    const counts = new Map();
    for (const object of mapData.objects || []) {
      const name = object.category_zh || object.category || 'Unknown';
      counts.set(name, (counts.get(name) || 0) + 1);
    }
    for (const [name, count] of counts) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.background = categoryColor(name);
      const label = document.createElement('span');
      label.textContent = `${name} ×${count}`;
      item.append(swatch, label);
      mapLegend.append(item);
    }
    const routeItem = document.createElement('span');
    routeItem.className = 'legend-item';
    routeItem.innerHTML = '<span class="legend-line route"></span><span>任务路线</span>';
    const trackItem = document.createElement('span');
    trackItem.className = 'legend-item';
    trackItem.innerHTML = '<span class="legend-line"></span><span>实际轨迹</span>';
    mapLegend.append(routeItem, trackItem);
    if (activeTask?.kind === 'find_object') {
      const candidate = document.createElement('span');
      candidate.className = 'legend-item';
      candidate.innerHTML = '<span style="color:#b56a00;font-weight:700">○</span><span>地图候选</span>';
      const miss = document.createElement('span');
      miss.className = 'legend-item';
      miss.innerHTML = '<span style="color:var(--danger);font-weight:700">×</span><span>已排除</span>';
      const match = document.createElement('span');
      match.className = 'legend-item';
      match.innerHTML = '<span style="color:var(--accent);font-weight:700">✓</span><span>已找到</span>';
      mapLegend.append(candidate, miss, match);
    }
    if (activeTask?.kind === 'patrol') {
      const baseline = document.createElement('span');
      baseline.className = 'legend-item';
      baseline.innerHTML = '<span style="color:#7b61a8;font-weight:700">●</span><span>视觉基线</span>';
      const normal = document.createElement('span');
      normal.className = 'legend-item';
      normal.innerHTML = '<span style="color:var(--accent);font-weight:700">✓</span><span>本轮正常</span>';
      const anomaly = document.createElement('span');
      anomaly.className = 'legend-item';
      anomaly.innerHTML = '<span style="color:var(--danger);font-weight:700">!</span><span>异常停车</span>';
      mapLegend.append(baseline, normal, anomaly);
    }
  }

  function standaloneDoorways(data) {
    const objects = data?.objects || [];
    const ids = new Set(objects.map(object => String(object.ann_id)));
    return (data?.doorways || []).filter(portal =>
      Array.isArray(portal.floor_xy) && portal.floor_xy.length >= 2 &&
      portal.floor_xy.slice(0, 2).every(value => typeof value === 'number' && Number.isFinite(value)) &&
      ![portal.navigation_ann_id, portal.anchor_ann_id, ...(portal.member_ids || [])].some(id => id != null && ids.has(String(id))));
  }

  function renderMap() {
    mapSvg.replaceChildren();
    if (!mapData?.objects?.length) return;
    mapLoading.classList.add('hidden');
    const width = 1000;
    const height = 720;
    const points = mapData.objects.map(object => object.nav_xy || object.floor_xy);
    points.push(...standaloneDoorways(mapData).map(portal => portal.floor_xy));
    for (const track of savedTracks) {
      if (!visibleTrackIds.has(String(track.id))) continue;
      for (const point of track.points || []) {
        if (Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1]))) points.push(point);
      }
    }
    points.push(...slamWorldCorners());
    if (robotData?.pose) points.push([Number(robotData.pose.x), Number(robotData.pose.y)]);
    let minX = Math.min(...points.map(point => Number(point[0])));
    let maxX = Math.max(...points.map(point => Number(point[0])));
    let minY = Math.min(...points.map(point => Number(point[1])));
    let maxY = Math.max(...points.map(point => Number(point[1])));
    const dataCenterX = (minX + maxX) / 2;
    const dataCenterY = (minY + maxY) / 2;
    const baseRangeX = Math.max(4, (maxX - minX) * 1.16);
    const baseRangeY = Math.max(4, (maxY - minY) * 1.16);
    const visibleX = baseRangeX / mapZoom;
    const visibleY = baseRangeY / mapZoom;
    mapVisibleWorld = { x: visibleX, y: visibleY };
    const followingRobot = robotData?.pose && (activeTask?.status === 'running' || activeTask?.status === 'canceling');
    const centerX = followingRobot ? Number(robotData.pose.x) : dataCenterX + mapPanX;
    const centerY = followingRobot ? Number(robotData.pose.y) : dataCenterY + mapPanY;
    minX = centerX - visibleX / 2; maxX = centerX + visibleX / 2;
    minY = centerY - visibleY / 2; maxY = centerY + visibleY / 2;
    const scale = Math.min((width - 70) / visibleX, (height - 70) / visibleY);
    const offsetX = (width - visibleX * scale) / 2;
    const offsetY = (height - visibleY * scale) / 2;
    const project = (x, y) => [offsetX + (x - minX) * scale, height - offsetY - (y - minY) * scale];
    mapSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const sceneLayer = svgNode('g', { transform: `rotate(${mapRotation} ${width / 2} ${height / 2})` });
    mapSvg.append(sceneLayer);

    if (slamVisible && slamData?.available && slamData.calibration) {
      const calibration = slamData.calibration;
      const resolution = Number(calibration.resolution);
      const angle = Number(calibration.rotation) * Math.PI / 180;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const imageHeightWorld = Number(slamData.height) * resolution;
      const topLeftWorld = [
        Number(calibration.origin_x) - sin * imageHeightWorld,
        Number(calibration.origin_y) + cos * imageHeightWorld,
      ];
      const [imageX, imageY] = project(topLeftWorld[0], topLeftWorld[1]);
      const matrix = [
        scale * resolution * cos,
        -scale * resolution * sin,
        scale * resolution * sin,
        scale * resolution * cos,
        imageX,
        imageY,
      ];
      sceneLayer.append(svgNode('image', {
        href: slamData.image_url,
        x: 0,
        y: 0,
        width: slamData.width,
        height: slamData.height,
        preserveAspectRatio: 'none',
        opacity: Number(calibration.opacity),
        transform: `matrix(${matrix.join(' ')})`,
        class: 'slam-map-image',
      }));
    }

    const taskTargets = new Set();
    const activeTargets = new Set();
    for (const step of activeTask?.steps || []) {
      const ids = (step.type === 'cruise' || step.type === 'patrol') ? (step.target_ann_ids || []) : [step.target_ann_id];
      for (const id of ids.filter(value => value !== undefined)) {
        if (activeTask?.kind !== 'find_object' || activeTask.candidate_status?.[String(id)] !== 'skipped') taskTargets.add(String(id));
      }
      if (step.index === activeTask?.current_step && activeTask?.current_target_ann_id == null) {
        for (const id of ids) activeTargets.add(String(id));
      }
    }
    if (activeTask?.current_target_ann_id != null) activeTargets.add(String(activeTask.current_target_ann_id));

    if (activeTask?.route_points?.length) {
      const route = [];
      if (robotData?.pose) route.push(project(robotData.pose.x, robotData.pose.y).join(','));
      const currentIndex = activeTask.current_step;
      let remaining = currentIndex === null || currentIndex === undefined
        ? activeTask.route_points
        : activeTask.route_points.filter(point => point.step_index >= currentIndex);
      if (activeTask.kind === 'find_object') {
        remaining = activeTask.route_points.filter(point => {
          const verdict = activeTask.candidate_status?.[String(point.ann_id)] || point.verdict;
          return verdict === 'candidate' || verdict === 'active';
        });
      } else if (activeTask.kind === 'patrol') {
        const currentId = String(activeTask.current_target_ann_id ?? '');
        const currentPosition = remaining.findIndex(point => String(point.ann_id) === currentId);
        if (currentPosition > 0) remaining = remaining.slice(currentPosition).concat(remaining.slice(0, currentPosition));
        if (remaining.length > 1) remaining = remaining.concat([remaining[0]]);
      }
      for (const point of remaining) route.push(project(point.x, point.y).join(','));
      const routeClass = activeTask.kind === 'find_object'
        ? 'task-route find-route'
        : (activeTask.kind === 'patrol' ? 'task-route patrol-route' : 'task-route');
      if (route.length > 1) sceneLayer.append(svgNode('polyline', { points: route.join(' '), class: routeClass }));
    }
    for (const savedTrack of savedTracks) {
      if (!visibleTrackIds.has(String(savedTrack.id)) || savedTrack.points?.length < 2) continue;
      const track = savedTrack.points
        .map(point => project(Number(point[0]), Number(point[1])).join(','))
        .join(' ');
      sceneLayer.append(svgNode('polyline', {
        points: track,
        class: 'saved-track',
        style: `--saved-track-color:${savedTrackColor(savedTrack)}`,
      }));
    }
    if (currentTrackVisible && robotData?.track?.length > 1) {
      const track = robotData.track.map(point => project(Number(point[0]), Number(point[1])).join(',')).join(' ');
      sceneLayer.append(svgNode('polyline', { points: track, class: 'robot-track' }));
    }

    const objectLayer = svgNode('g');
    for (const object of mapData.objects) {
      const [x, y] = (object.nav_xy || object.floor_xy).map(Number);
      const [cx, cy] = project(x, y);
      const symbolSize = 36;
      const haloRadius = symbolSize * .68;
      const name = object.category_zh || object.category || '目标';
      const kind = objectSymbolKind(object);
      const classes = ['map-object'];
      if (String(object.ann_id) === String(selectedObjectId)) classes.push('selected');
      if (capabilitySelect.value === 'patrol' && patrolSelectedIds.includes(object.ann_id)) classes.push('patrol-selected');
      if (String(object.ann_id) === String(welcomePickupAnnId)) classes.push('welcome-pickup');
      if (String(object.ann_id) === String(welcomeReturnAnnId)) classes.push('welcome-return');
      if (taskTargets.has(String(object.ann_id))) classes.push('target');
      if (activeTargets.has(String(object.ann_id))) classes.push('active-target');
      if (mapQueryIds.has(String(object.ann_id))) classes.push('query-target');
      const findVerdict = activeTask?.candidate_status?.[String(object.ann_id)];
      if (activeTask?.kind === 'find_object' && findVerdict === 'candidate') classes.push('find-candidate');
      if (activeTask?.kind === 'find_object' && findVerdict === 'miss') classes.push('checked-miss');
      if (activeTask?.kind === 'find_object' && findVerdict === 'match') classes.push('matched-target');
      const patrolVerdict = activeTask?.patrol_status_by_ann?.[String(object.ann_id)];
      if (activeTask?.kind === 'patrol' && patrolVerdict === 'baseline') classes.push('patrol-baseline');
      if (activeTask?.kind === 'patrol' && patrolVerdict === 'normal') classes.push('patrol-normal');
      if (activeTask?.kind === 'patrol' && patrolVerdict === 'anomaly') classes.push('patrol-anomaly');
      const color = categoryColor(name);
      const group = svgNode('g', {
        class: classes.join(' '),
        'data-id': object.ann_id,
        style: `--object-color:${color}`,
        transform: `rotate(${-mapRotation} ${cx} ${cy})`,
        role: 'button',
        tabindex: 0,
        'aria-label': `${name} #${object.ann_id}`,
      });
      group.append(
        svgNode('circle', { cx, cy, r: haloRadius + 5, class: 'object-hit' }),
        svgNode('circle', { cx, cy, r: haloRadius, class: 'object-halo' }),
      );
      appendObjectIcon(group, kind, cx, cy, symbolSize);
      let verdictSymbol = '';
      let verdictClass = '';
      if (findVerdict === 'miss') { verdictSymbol = '×'; verdictClass = 'miss'; }
      if (findVerdict === 'uncertain') { verdictSymbol = '?'; verdictClass = 'uncertain'; }
      if (findVerdict === 'match') { verdictSymbol = '✓'; verdictClass = 'match'; }
      if (patrolVerdict === 'normal') { verdictSymbol = '✓'; verdictClass = 'normal'; }
      if (patrolVerdict === 'anomaly') { verdictSymbol = '!'; verdictClass = 'anomaly'; }
      if (verdictSymbol) {
        const verdict = svgNode('text', {
          x: cx + haloRadius * .72, y: cy - haloRadius * .72,
          class: `map-verdict ${verdictClass}`,
        });
        verdict.textContent = verdictSymbol;
        group.append(verdict);
      }
      const tooltip = svgNode('title');
      tooltip.textContent = `${name} #${object.ann_id} · (${x.toFixed(2)}, ${y.toFixed(2)})`;
      group.append(tooltip);
      group.addEventListener('click', () => {
        if (suppressMapClick) return;
        handleMapObjectSelection(object);
      });
      group.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        handleMapObjectSelection(object);
      });
      objectLayer.append(group);
    }
    sceneLayer.append(objectLayer);

    // Unanchored visual doorways retain their portal IDs. They are display
    // evidence, not invented ann_ids or verified robot navigation targets.
    for (const portal of standaloneDoorways(mapData)) {
      const [cx, cy] = project(...portal.floor_xy);
      const group = svgNode('g', { class: 'map-portal', 'data-portal-id': portal.portal_id,
        transform: `rotate(${-mapRotation} ${cx} ${cy})`, role: 'img',
        'aria-label': `${portal.category_zh}（门洞记录，通行未验证）` });
      group.append(svgNode('circle', { cx, cy, r: 24, fill: '#8374cb', stroke: '#fff', 'stroke-width': 3 }));
      appendObjectIcon(group, 'door', cx, cy, 36);
      const title = svgNode('title');
      title.textContent = `${portal.category_zh} · ${portal.portal_id} · 仅显示已保存门洞，通行与导航点未验证`;
      group.append(title);
      sceneLayer.append(group);
    }

    if (robotData?.pose) {
      const [rx, ry] = project(robotData.pose.x, robotData.pose.y);
      const angle = -Number(robotData.pose.theta || 0) * 180 / Math.PI + 90;
      const marker = svgNode('polygon', {
        points: `${rx},${ry - 28} ${rx - 19},${ry + 21} ${rx},${ry + 12} ${rx + 19},${ry + 21}`,
        class: 'robot-marker', transform: `rotate(${angle} ${rx} ${ry})`,
      });
      const title = svgNode('title');
      title.textContent = `机器人 (${Number(robotData.pose.x).toFixed(2)}, ${Number(robotData.pose.y).toFixed(2)})`;
      marker.append(title);
      sceneLayer.append(marker);
    }
    sync3DScene();
  }

  function showSlamCalibration(value) {
    const calibration = value?.calibration;
    slamName.textContent = value?.available ? `${value.name} · ${value.width}×${value.height}` : '未加载';
    if (!calibration) return;
    slamOriginX.value = Number(calibration.origin_x).toFixed(3);
    slamOriginY.value = Number(calibration.origin_y).toFixed(3);
    slamResolution.value = Number(calibration.resolution).toFixed(5);
    slamRotation.value = Number(calibration.rotation).toFixed(1);
    slamOpacity.value = String(calibration.opacity);
  }

  function readSlamCalibration() {
    return {
      origin_x: Number(slamOriginX.value),
      origin_y: Number(slamOriginY.value),
      resolution: Number(slamResolution.value),
      rotation: Number(slamRotation.value),
      opacity: Number(slamOpacity.value),
    };
  }

  function slamWorldCorners() {
    if (!slamVisible || !slamData?.available || !slamData.calibration) return [];
    const calibration = slamData.calibration;
    const angle = Number(calibration.rotation) * Math.PI / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const width = Number(slamData.width) * Number(calibration.resolution);
    const height = Number(slamData.height) * Number(calibration.resolution);
    return [[0, 0], [width, 0], [0, height], [width, height]].map(([x, y]) => [
      Number(calibration.origin_x) + cos * x - sin * y,
      Number(calibration.origin_y) + sin * x + cos * y,
    ]);
  }

  async function loadSlamMap() {
    slamData = await api('/api/slam');
    showSlamCalibration(slamData);
    renderMap();
    if (activeTask?.kind === 'find_object') renderMessages();
    renderMappingMap();
  }

  function fileAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.addEventListener('load', () => resolve(String(reader.result).split(',', 2)[1] || ''));
      reader.addEventListener('error', () => reject(new Error('读取 SLAM 图片失败')));
      reader.readAsDataURL(file);
    });
  }

  const REFERENCE_MAX_EDGE = 1024;
  const REFERENCE_UPLOAD_TARGET_BYTES = Math.floor(700 * 1024);
  const REFERENCE_SOURCE_MAX_BYTES = 20 * 1024 * 1024;

  function canvasAsJpeg(canvas, quality) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('参考图片压缩失败')), 'image/jpeg', quality);
    });
  }

  async function optimizeReferenceFile(file) {
    if (file.size <= 0 || file.size > REFERENCE_SOURCE_MAX_BYTES) {
      throw new Error('参考原图必须小于 20 MB');
    }
    const sourceUrl = URL.createObjectURL(file);
    const image = new Image();
    try {
      await new Promise((resolve, reject) => {
        image.addEventListener('load', resolve, { once: true });
        image.addEventListener('error', () => reject(new Error('无法读取参考图片')), { once: true });
        image.src = sourceUrl;
      });
      const longestEdge = Math.max(image.naturalWidth, image.naturalHeight);
      if (!longestEdge) throw new Error('参考图片尺寸无效');
      let scale = Math.min(1, REFERENCE_MAX_EDGE / longestEdge);
      for (let attempt = 0; attempt < 4; attempt += 1) {
        const width = Math.max(1, Math.round(image.naturalWidth * scale));
        const height = Math.max(1, Math.round(image.naturalHeight * scale));
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        context.fillStyle = '#fff';
        context.fillRect(0, 0, width, height);
        context.drawImage(image, 0, 0, width, height);
        for (const quality of [0.84, 0.74, 0.64]) {
          const blob = await canvasAsJpeg(canvas, quality);
          if (blob.size <= REFERENCE_UPLOAD_TARGET_BYTES) {
            const stem = file.name.replace(/\.[^.]+$/, '') || 'reference';
            return new File([blob], `${stem}.jpg`, { type: 'image/jpeg' });
          }
        }
        scale *= 0.72;
      }
      throw new Error('参考图片无法压缩到 700 KB 内，请换一张更清晰的近景照片');
    } finally {
      URL.revokeObjectURL(sourceUrl);
    }
  }

  function clearPendingReference() {
    if (pendingReference?.previewUrl) URL.revokeObjectURL(pendingReference.previewUrl);
    pendingReference = null;
    referenceFileInput.value = '';
    referencePreview.hidden = true;
    referencePreviewImage.removeAttribute('src');
    referencePreviewName.textContent = '';
  }

  function selectReferenceFile(file) {
    if (!file) return;
    const allowed = new Set(['image/png', 'image/jpeg', 'image/webp']);
    if (!allowed.has(file.type)) throw new Error('参考图片仅支持 PNG、JPEG 或 WebP');
    if (file.size <= 0 || file.size > REFERENCE_SOURCE_MAX_BYTES) throw new Error('参考原图必须小于 20 MB');
    clearPendingReference();
    const previewUrl = URL.createObjectURL(file);
    const selected = { originalFile: file, file: null, name: file.name, previewUrl, optimizedPromise: null };
    pendingReference = selected;
    referencePreviewImage.src = previewUrl;
    referencePreviewName.textContent = `${file.name}（已显示，后台优化中）`;
    referencePreview.hidden = false;
    selected.optimizedPromise = optimizeReferenceFile(file).then(optimized => {
      selected.file = optimized;
      selected.name = optimized.name;
      if (pendingReference === selected) {
        referencePreviewName.textContent = `${file.name} → ${optimized.name}（${Math.ceil(optimized.size / 1024)} KB）`;
        if (!sending) composerStatus.textContent = '参考图已就绪';
      }
      return optimized;
    });
    selected.optimizedPromise.catch(error => {
      if (pendingReference === selected) composerStatus.textContent = error.message || '参考图优化失败';
    });
    return selected;
  }

  async function loadMapCatalog() {
    const catalog = await api('/api/maps');
    mapSelect.replaceChildren();
    for (const name of catalog.files || []) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      option.selected = name === catalog.current;
      mapSelect.append(option);
    }
    await loadCurrentMap();
  }

  async function loadCurrentMap() {
    mapLoading.classList.remove('hidden');
    mapData = await api('/api/map');
    selectedObjectId = null;
    welcomePickupAnnId = null;
    welcomeReturnAnnId = null;
    welcomeSelectionRole = capabilitySelect.value === 'welcome' ? 'pickup' : null;
    mapZoom = DEFAULT_MAP_ZOOM;
    mapPanX = 0;
    mapPanY = 0;
    renderMapLegend();
    renderObjectInspector();
    refreshWelcomeSetup();
    renderMap();
  }

  function syncTask(task) {
    if (!task || !task.conversation_id || task.conversation_id === activeId) activeTask = task;
    if (!task) return false;
    let changed = false;
    for (const conversation of conversations) {
      for (const message of conversation.messages) {
        if (task.conversation_id && task.conversation_id !== conversation.id) continue;
        if (message.taskId === task.id) {
          const before = JSON.stringify(message.task || {});
          message.task = task;
          if (before !== JSON.stringify(task)) changed = true;
        }
      }
    }
    if (changed) {
      saveConversations();
      renderMapLegend();
    }
    return changed;
  }

  async function pollRobotStatus() {
    let delay = 2600;
    try {
      const polledConversation = activeId;
      const status = await api(`/api/robot/status?conversation_id=${encodeURIComponent(polledConversation || '')}`);
      const previousBlocker = executionBlocker(robotData);
      robotData = status.robot;
      const safetyChanged = previousBlocker !== executionBlocker(robotData);
      const changed = activeId === polledConversation ? syncTask(status.task) : false;
      const dot = document.createElement('span');
      dot.className = 'status-dot ' + (robotData?.online ? 'online' : 'offline');
      const label = document.createElement('span');
      if (robotData?.online && robotData.pose) {
        let warning = '';
        if (mapData?.objects?.length) {
          const nearest = Math.min(...mapData.objects.map(object => {
            const point = object.nav_xy || object.floor_xy;
            return Math.hypot(Number(point[0]) - Number(robotData.pose.x), Number(point[1]) - Number(robotData.pose.y));
          }));
          if (nearest > 15) warning = ' · 坐标系可能未对齐';
        }
        label.textContent = `在线 · (${Number(robotData.pose.x).toFixed(2)}, ${Number(robotData.pose.y).toFixed(2)}) · ${robotData.move_status || 'idle'}${warning}${executionBlocker(robotData) ? ' · ' + executionBlocker(robotData) : ''}`;
      } else {
        label.textContent = robotData?.error ? '底盘离线' : '状态未知';
      }
      robotStateText.replaceChildren(dot, label);
      const mappingDot = dot.cloneNode(true);
      const mappingLabel = label.cloneNode(true);
      mappingRobotState.replaceChildren(mappingDot, mappingLabel);
      renderMap();
      if (workspaceMode === 'mapping') renderMappingMap();
      if (changed || safetyChanged || (activeTask?.kind === 'find_object' && ['running', 'canceling'].includes(activeTask.status))) renderMessages();
      if (status.task?.status === 'running' || status.task?.status === 'canceling') delay = 700;
    } catch (_) {
      robotData = { online: false };
      renderMessages();
      robotStateText.textContent = '无法读取机器人状态';
    }
    setTimeout(pollRobotStatus, delay);
  }

  async function executeTask(taskId) {
    try {
      const result = await api('/api/task/execute', { task_id: taskId, conversation_id: activeId });
      syncTask(result.task);
      renderMessages();
      renderMap();
    } catch (error) {
      composerStatus.textContent = error.message;
    }
  }

  async function confirmWelcomeGuest(task, confirmationId, accepted) {
    const conversationId = task.conversation_id || activeId;
    try {
      const result = await api('/api/task/welcome-confirm', {
        task_id: task.id, conversation_id: conversationId,
        confirmation_id: confirmationId, accepted,
      });
      syncTask(result.task);
      renderMessages();
      renderMap();
    } catch (error) {
      composerStatus.textContent = error.message;
      renderMessages();
    }
  }

  async function cancelTask(taskId) {
    try {
      const result = await api('/api/task/cancel', { task_id: taskId, conversation_id: activeId });
      syncTask(result.task);
      renderMessages();
      renderMap();
    } catch (error) {
      composerStatus.textContent = error.message;
    }
  }

  function mappingPayload() {
    return {
      max_points: Number(mappingMaxPoints.value),
      point_spacing: Number(mappingSpacing.value),
      heading_step_deg: Number(mappingHeadingStep.value),
      robot_radius: Number(mappingRobotRadius.value),
      safety_margin: Number(mappingSafetyMargin.value),
      settle_time: Number(mappingSettleTime.value),
    };
  }

  async function createMappingPlan() {
    mappingPlanButton.disabled = true;
    mappingHeaderStatus.textContent = '正在分析 SLAM 地图并计算路线';
    try {
      const result = await api('/api/mapping/plan', mappingPayload());
      selectedMapping = result.session;
      selectedMappingId = result.session.id;
      await loadMappingSessions();
    } catch (error) {
      mappingHeaderStatus.textContent = error.message || '路线计算失败';
    } finally {
      mappingPlanButton.disabled = false;
      renderMappingWorkspace();
    }
  }

  async function startMapping(resume = false) {
    if (!selectedMappingId) return;
    const path = resume ? '/api/mapping/resume' : '/api/mapping/start';
    try {
      const result = await api(path, { session_id: selectedMappingId });
      selectedMapping = result.session;
      renderMappingWorkspace();
    } catch (error) {
      mappingHeaderStatus.textContent = error.message;
    }
  }

  async function stopMapping() {
    if (!selectedMappingId) return;
    try {
      const result = await api('/api/mapping/cancel', { session_id: selectedMappingId });
      selectedMapping = result.session;
      renderMappingWorkspace();
    } catch (error) {
      mappingHeaderStatus.textContent = error.message;
    }
  }

  async function applyMappingResult() {
    if (!selectedMappingId) return;
    try {
      const result = await api('/api/mapping/apply', { session_id: selectedMappingId });
      mapData = result.map;
      await switchWorkspace('function');
      openMapPanel();
      await loadMapCatalog();
      renderMapLegend();
      renderMap();
    } catch (error) {
      mappingHeaderStatus.textContent = error.message;
    }
  }

  async function pollMappingStatus() {
    let delay = workspaceMode === 'mapping' ? 900 : 2600;
    try {
      if (selectedMappingId) {
        const result = await api(`/api/mapping/status?session_id=${encodeURIComponent(selectedMappingId)}`);
        if (result.session) {
          const oldStatus = selectedMapping?.status;
          selectedMapping = result.session;
          renderMappingWorkspace();
          if (oldStatus !== selectedMapping.status || ['running', 'uploading', 'processing', 'server_processing', 'canceling'].includes(selectedMapping.status)) {
            const catalog = await api('/api/mapping/sessions');
            mappingSessions = catalog.sessions || [];
            renderHistory();
          }
        }
      } else if (workspaceMode === 'mapping') {
        await loadMappingSessions();
      }
    } catch (error) {
      if (workspaceMode === 'mapping') mappingHeaderStatus.textContent = `状态读取失败：${error.message}`;
    }
    setTimeout(pollMappingStatus, delay);
  }

  function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 150) + 'px';
  }

  async function api(path, payload = null) {
    const options = payload === null ? {} : (payload instanceof Blob
      ? {
        method: 'POST',
        headers: { 'Content-Type': payload.type || 'application/octet-stream' },
        body: payload
      }
      : {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    const controller = new AbortController();
    const timeoutMs = path === '/api/health'
      ? 5000
      : (path.startsWith('/api/reference/upload') ? 300000 : 120000);
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal });
      let data = {};
      try { data = await response.json(); } catch (_) {}
      if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
      return data;
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw new Error(path === '/api/reference/upload'
          ? '上传参考图超时，请确认机器人 Web 服务仍在运行后重试'
          : '机器人响应超时，请确认服务连接后重试');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  function refreshActionAvailability() {
    refreshPatrolSetup();
    const speechCanToggle = speechState === 'idle' || speechState === 'recording';
    sendButton.disabled = sending || !backendOnline || !conversationStoreReady || speechState !== 'idle';
    micButton.disabled = sending || !backendOnline || !conversationStoreReady || !speechCanToggle;
  }

  function setSpeechState(nextState, statusText = '') {
    speechState = nextState;
    const recording = nextState === 'recording';
    micButton.classList.toggle('listening', recording);
    micButton.setAttribute('aria-label', recording ? '停止录音' : '语音输入');
    micButton.dataset.tip = recording ? '停止录音' : '语音输入';
    cancelRecordingButton.hidden = !recording;
    if (statusText) composerStatus.textContent = statusText;
    refreshActionAvailability();
  }

  async function agentChat(payload, onEvent) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 240000);
    let reader;
    try {
      const response = await fetch('/api/agent/chat', {
        method: 'POST', cache: 'no-store', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error('对话服务暂时不可用，请稍后重试');
      let result = null;
      const consume = line => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'error') throw new Error(event.text);
        if (event.type === 'final') result = event;
        else onEvent(event);
      };
      if (!response.body?.getReader) {
        (await response.text()).split('\n').forEach(consume);
      } else {
        reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          lines.forEach(consume);
          if (done) { consume(buffer); break; }
        }
      }
      if (!result) throw new Error('连接中断，未收到完整结果；请检查任务面板后重试');
      return result;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('对话处理超时，请稍后重试');
      throw error;
    } finally {
      clearTimeout(timer);
      if (reader) { try { await reader.cancel(); } catch (_) {} reader.releaseLock(); }
    }
  }

  async function sendMessage(forcedText = '', targetSelection = null) {
    let patrolRequest = null;
    if (capabilitySelect.value === 'patrol') {
      try { patrolRequest = selectedPatrolRequest(); }
      catch (error) { composerStatus.textContent = patrolSelectionHint.textContent = error.message; openMapPanel(); return; }
    }
    const question = patrolRequest ? patrolRequest.instruction : (forcedText || input.value).trim();
    if (!question || sending || !conversationStoreReady) return;
    if (!backendOnline) {
      composerStatus.textContent = '机器人连接不可用，请稍后重试';
      return;
    }
    const welcomeRequest = capabilitySelect.value === 'welcome';
    const selectedWelcomePickup = welcomePickupAnnId;
    const selectedWelcomeReturn = welcomeReturnAnnId;
    if (welcomeRequest && !pendingReference) {
      composerStatus.textContent = '请先上传目标人物的上半身参考照片';
      return;
    }
    if (welcomeRequest && (selectedWelcomePickup == null || selectedWelcomeReturn == null)) {
      composerStatus.textContent = '请先在地图中分别选择接人点和返回点';
      openMapPanel();
      if (selectedWelcomePickup == null) beginWelcomeLocationSelection('pickup');
      else beginWelcomeLocationSelection('return');
      return;
    }
    if (welcomeRequest && String(selectedWelcomePickup) === String(selectedWelcomeReturn)) {
      composerStatus.textContent = '接人点和返回点不能相同';
      return;
    }
    const conversation = ensureConversation();
    const conversationId = conversation.id;
    const assistantId = uid();
    const userId = uid();
    pendingTurn = { conversationId, userId, assistantId };
    const selectedReference = patrolRequest ? null : pendingReference;
    const recentHistory = conversation.messages
      .filter(message => message.text && message.state !== 'error')
      .slice(-16)
      .map(message => ({
        role: message.role, text: message.text,
        capture_id: message.captureId || undefined
      }));
    const localReferenceUrl = selectedReference?.originalFile ? URL.createObjectURL(selectedReference.originalFile) : '';
    conversation.messages.push({ id: userId, role: 'user', text: question, referenceImage: localReferenceUrl, createdAt: Date.now() });
    conversation.messages.push({ id: assistantId, role: 'assistant', text: '', image: '', state: 'routing', createdAt: Date.now() });
    if (conversation.title === '新对话') conversation.title = question.slice(0, 24);
    input.value = '';
    autoResize();
    sending = true;
    refreshActionAvailability();
    composerStatus.textContent = selectedReference ? '已显示参考图，正在同步上传' : '小卡正在思考';
    if (pendingReference === selectedReference) clearPendingReference();
    saveConversations();
    renderAll();

    const locate = () => {
      const targetConversation = conversations.find(item => item.id === conversationId);
      return targetConversation?.messages.find(item => item.id === assistantId);
    };

    try {
      const uploadPromise = selectedReference
        ? selectedReference.optimizedPromise.then(file => api(
          `/api/reference/upload?name=${encodeURIComponent(file.name)}&conversation_id=${encodeURIComponent(conversationId)}&turn_id=${encodeURIComponent(userId)}`,
          file
        ))
        : Promise.resolve(null);
      const reference = await uploadPromise;
      if (reference) {
        const user = conversation.messages.find(message => message.id === userId);
        if (user) user.referenceStoredImage = reference.image_url;
        saveConversations();
      }
      const assistant = locate();
      if (!assistant) return;
      if (patrolRequest) {
        const planned = await api('/api/task/patrol/plan', {
          ...patrolRequest, conversation_id: conversationId,
          user_message_id: userId, assistant_message_id: assistantId,
        });
        assistant.taskId = planned.task.id;
        assistant.task = planned.task;
        assistant.text = '巡逻计划已生成，请核对地点及顺序后点击开始执行。';
        assistant.state = 'done';
        if (activeId === conversationId) { activeTask = planned.task; closeMapPanel(); }
        renderMap();
        return;
      }
      if (welcomeRequest) {
        openMapPanel();
        if (!reference) throw new Error('请先上传目标人物的参考照片');
        mapQueryIds = new Set(
          [selectedWelcomePickup, selectedWelcomeReturn].map(String)
        );
        assistant.state = 'planning';
        composerStatus.textContent = '正在生成迎宾任务';
        saveConversations();
        if (activeId === conversationId) renderMessages();
        const planned = await api('/api/task/welcome/plan', {
          conversation_id: conversationId, user_message_id: userId, assistant_message_id: assistantId,
          instruction: question,
          reference,
          pickup_ann_id: selectedWelcomePickup,
          return_ann_id: selectedWelcomeReturn
        });
        assistant.taskId = planned.task.id;
        assistant.task = planned.task;
        assistant.text = planned.task.understanding || '';
        assistant.state = 'done';
        if (activeId === conversationId) activeTask = planned.task;
        renderMap();
        return;
      }
      assistant.state = 'thinking';
      assistant.agentProgress = '小卡正在思考';
      const result = await agentChat({ question, history: recentHistory, reference, conversation_id: conversationId,
        user_message_id: userId, assistant_message_id: assistantId,
        ...(targetSelection ? { target_selection: targetSelection } : {}) }, event => {
        if (event.text) {
          assistant.agentProgress = event.text;
          composerStatus.textContent = event.text;
        }
        if (event.type === 'image') {
          assistant.image = event.image_url;
          assistant.captureId = event.capture_id;
          assistant.imageSource = event.image_source;
        }
        if (activeId === conversationId) renderMessages();
      });
      assistant.text = result.text;
      assistant.toolTrace = result.tool_trace || [];
      assistant.execution = result.execution;
      assistant.responseKind = result.response_kind;
      assistant.incomplete = Boolean(result.incomplete);
      assistant.targetChoices = result.target_choices;
      assistant.targetChoiceSnapshot = result.target_choice_snapshot;
      assistant.image = result.image_url || assistant.image;
      assistant.captureId = result.capture_id || assistant.captureId;
      assistant.imageSource = result.image_source || assistant.imageSource;
      if (activeId === conversationId) mapQueryIds = new Set((result.target_ids || []).map(String));
      if (result.task) {
        assistant.taskId = result.task.id;
        assistant.task = result.task;
        if (activeId === conversationId) {
          activeTask = result.task;
          mapQueryIds = new Set((result.task.clarification?.candidate_ann_ids || []).map(String));
          openMapPanel();
        }
      }
      assistant.state = 'done';
      renderMap();
    } catch (error) {
      const assistant = locate();
      if (assistant) {
        assistant.text = userFacingError(error.message || '请求失败');
        assistant.state = 'error';
      }
    } finally {
      sending = false;
      pendingTurn = null;
      refreshActionAvailability();
      if (activeId === conversationId) capabilitySelect.value = '';
      capabilityHint.textContent = '选择后会再次填入可修改的示例指令';
      welcomeSelectionRole = null;
      refreshWelcomeSetup();
      referenceButton.setAttribute('aria-label', '上传寻物参考图');
      refreshPatrolSetup();
      referenceButton.dataset.tip = '上传寻物参考图';
      composerStatus.textContent = backendOnline ? '就绪' : '机器人连接不可用';
      saveConversations();
      if (activeId === conversationId) renderMessages();
      // Backend owns the final record, including tasks completed after a tab switch.
      try { await loadServerConversation(conversationId); } catch (_) { /* local draft remains recoverable */ }
      input.focus();
    }
  }

  function supportedRecordingMimeType() {
    if (!window.MediaRecorder) return '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/mp4;codecs=mp4a.40.2',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      'audio/webm'
    ];
    return candidates.find(type => !MediaRecorder.isTypeSupported || MediaRecorder.isTypeSupported(type)) || '';
  }

  function clearRecordingTimers() {
    clearInterval(recordingTimer);
    clearTimeout(recordingLimitTimer);
    recordingTimer = null;
    recordingLimitTimer = null;
  }

  function releaseMicrophone() {
    clearRecordingTimers();
    microphoneStream?.getTracks().forEach(track => track.stop());
    microphoneStream = null;
  }

  function updateRecordingClock() {
    const elapsed = Math.min(MAX_RECORDING_MS, Date.now() - recordingStartedAt);
    const seconds = Math.ceil(elapsed / 1000);
    composerStatus.textContent = `正在录音 ${seconds} / ${MAX_RECORDING_MS / 1000} 秒，再次点击麦克风结束`;
  }

  async function startPhoneRecording() {
    if (sending || speechState !== 'idle') return;
    if (!window.isSecureContext) {
      composerStatus.textContent = '手机麦克风需要通过 HTTPS 打开本页面';
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      composerStatus.textContent = '当前浏览器不支持网页录音，请使用最新版 Chrome 或 Safari';
      return;
    }

    setSpeechState('requesting', '正在请求麦克风权限');
    try {
      microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      if (!backendOnline) throw new Error('机器人连接已断开');
      recordingMimeType = supportedRecordingMimeType();
      const options = recordingMimeType ? { mimeType: recordingMimeType } : undefined;
      mediaRecorder = options
        ? new MediaRecorder(microphoneStream, options)
        : new MediaRecorder(microphoneStream);
      recordingChunks = [];
      recordingCancelled = false;
      mediaRecorder.addEventListener('dataavailable', event => {
        if (event.data?.size) recordingChunks.push(event.data);
      });
      mediaRecorder.addEventListener('stop', finishPhoneRecording, { once: true });
      mediaRecorder.addEventListener('error', event => {
        recordingCancelled = true;
        releaseMicrophone();
        setSpeechState('idle', event.error?.message || '录音失败，请重试');
      }, { once: true });
      mediaRecorder.start(250);
      recordingStartedAt = Date.now();
      setSpeechState('recording');
      updateRecordingClock();
      recordingTimer = setInterval(updateRecordingClock, 500);
      recordingLimitTimer = setTimeout(() => stopPhoneRecording(false), MAX_RECORDING_MS);
    } catch (error) {
      releaseMicrophone();
      const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError';
      setSpeechState('idle', denied
        ? '麦克风权限被拒绝，请在浏览器设置中允许后重试'
        : (error.message || '无法启动手机麦克风'));
    }
  }

  function stopPhoneRecording(cancelled = false) {
    if (speechState !== 'recording') return;
    recordingCancelled = cancelled;
    clearRecordingTimers();
    if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
    releaseMicrophone();
    setSpeechState(cancelled ? 'idle' : 'uploading', cancelled ? '已取消录音' : '正在上传录音');
  }

  async function finishPhoneRecording() {
    const cancelled = recordingCancelled;
    const chunks = recordingChunks;
    const mimeType = mediaRecorder?.mimeType || recordingMimeType || chunks[0]?.type || 'audio/webm';
    mediaRecorder = null;
    recordingChunks = [];
    if (cancelled) return;

    try {
      const blob = new Blob(chunks, { type: mimeType });
      if (!blob.size) throw new Error('没有录到声音，请重试');
      if (blob.size > MAX_AUDIO_BYTES) throw new Error('录音超过 8 MB，请缩短后重试');
      setSpeechState('transcribing', '正在识别语音');
      const result = await api('/api/audio/transcribe', blob);
      const text = String(result.text || '').trim();
      if (!text) throw new Error('没有识别到有效语音');
      input.value = text;
      autoResize();
      setSpeechState('idle', autoSendVoice.checked ? '语音已识别，正在发送' : '语音已转成文字，可修改后发送');
      if (autoSendVoice.checked) {
        await sendMessage(text);
      } else {
        input.focus();
      }
    } catch (error) {
      setSpeechState('idle', error.message || '语音识别失败，请重试');
    }
  }

  function togglePhoneRecording() {
    if (speechState === 'recording') stopPhoneRecording(false);
    else startPhoneRecording();
  }

  function openSidebar() { sidebar.classList.add('open'); backdrop.classList.add('open'); }
  function closeSidebar() { sidebar.classList.remove('open'); backdrop.classList.remove('open'); }
  function openMapPanel() {
    app.classList.add('is-map-open');
    mapPanel.classList.add('open');
    document.getElementById('mapButton').setAttribute('aria-expanded', 'true');
    renderMap();
  }
  function closeMapPanel() {
    app.classList.remove('is-map-open');
    mapPanel.classList.remove('open');
    mapTrackPanel.classList.remove('open');
    trackPanelToggleButton.classList.remove('active');
    document.getElementById('mapButton').setAttribute('aria-expanded', 'false');
  }
  function toggleMapPanel() {
    if (app.classList.contains('is-map-open')) closeMapPanel();
    else openMapPanel();
  }

  function applyCapabilityExample() {
    const example = capabilityExamples[capabilitySelect.value];
    const welcomeEnabled = capabilitySelect.value === 'welcome';
    referenceButton.setAttribute(
      'aria-label',
      welcomeEnabled ? '上传迎宾人物参考图' : '上传寻物参考图'
    );
    referenceButton.dataset.tip = welcomeEnabled ? '上传迎宾人物参考图' : '上传寻物参考图';
    if (welcomeEnabled && welcomePickupAnnId == null) welcomeSelectionRole = 'pickup';
    if (!welcomeEnabled) welcomeSelectionRole = null;
    refreshWelcomeSetup();
    refreshPatrolSetup();
    renderMap();
    if (capabilitySelect.value === 'patrol') {
      input.value = '';
      capabilityHint.textContent = example.hint;
      openMapPanel();
      autoResize();
      return;
    }
    if (!example) {
      capabilityHint.textContent = '选择后会填入可修改的示例指令';
      return;
    }
    input.value = example.prompt;
    capabilityHint.textContent = example.hint;
    autoResize();
    input.focus();
    const placeholder = input.value.match(/\[[^\]]+\]/);
    if (placeholder) input.setSelectionRange(placeholder.index, placeholder.index + placeholder[0].length);
    if (capabilitySelect.value === 'find_object' && !pendingReference) {
      composerStatus.textContent = '请点击输入框右侧的图片按钮上传寻物参考图';
    }
    if (welcomeEnabled) {
      openMapPanel();
      if (!pendingReference) {
        composerStatus.textContent = '请上传人物参考图，并在地图选择接人点和返回点';
      }
    }
  }

  async function checkHealth() {
    if (!navigator.onLine) {
      backendOnline = false;
      connectionDot.className = 'status-dot offline';
      connectionText.textContent = '网络已断开';
      mobileConnectionDot.className = 'status-dot offline';
      mobileConnectionText.textContent = '离线';
      if (!sending && speechState === 'idle') composerStatus.textContent = '网络已断开，暂时无法发送';
      refreshActionAvailability();
      return;
    }
    try {
      const health = await api('/api/health');
      backendOnline = true;
      if (!conversationStoreReady) await initializeConversationStore();
      connectionDot.className = 'status-dot online';
      connectionText.textContent = health.replay ? '任务回放' : health.mock ? '模拟模式' : '机器人在线';
      mobileConnectionDot.className = 'status-dot online';
      mobileConnectionText.textContent = health.replay ? '回放' : health.mock ? '模拟' : '在线';
      if (health.replay && !document.getElementById('replayEntryLink')) {
        const replayLink = document.createElement('a');
        replayLink.id = 'replayEntryLink'; replayLink.href = '/replay';
        replayLink.textContent = '打开 Agent 与机器人任务回放';
        document.getElementById('emptyState').append(replayLink);
      }
      if (!sending && speechState === 'idle' && conversationStoreReady) composerStatus.textContent = '就绪';
    } catch (_) {
      backendOnline = false;
      connectionDot.className = 'status-dot offline';
      connectionText.textContent = '连接失败';
      mobileConnectionDot.className = 'status-dot offline';
      mobileConnectionText.textContent = '断开';
      if (!sending && speechState === 'idle') composerStatus.textContent = '机器人连接失败，正在等待恢复';
    }
    refreshActionAvailability();
  }

  composer.addEventListener('submit', event => { event.preventDefault(); sendMessage(); });
  input.addEventListener('input', autoResize);
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  micButton.addEventListener('click', togglePhoneRecording);
  cancelRecordingButton.addEventListener('click', () => stopPhoneRecording(true));
  autoSendVoice.addEventListener('change', () => {
    localStorage.setItem(AUTO_SEND_VOICE_KEY, String(autoSendVoice.checked));
    composerStatus.textContent = autoSendVoice.checked ? '已开启语音识别后自动发送' : '语音识别后将等待确认';
  });
  window.addEventListener('online', checkHealth);
  window.addEventListener('offline', () => {
    if (speechState === 'recording') stopPhoneRecording(true);
    checkHealth();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && speechState === 'recording') stopPhoneRecording(true);
  });
  window.addEventListener('pagehide', releaseMicrophone);
  referenceButton.addEventListener('click', () => referenceFileInput.click());
  selectWelcomePickupButton.addEventListener('click', () => beginWelcomeLocationSelection('pickup'));
  selectWelcomeReturnButton.addEventListener('click', () => beginWelcomeLocationSelection('return'));
  removeReferenceButton.addEventListener('click', clearPendingReference);
  referenceFileInput.addEventListener('change', async () => {
    const file = referenceFileInput.files?.[0];
    if (!file) return;
    try {
      selectReferenceFile(file);
      composerStatus.textContent = '参考图已显示，正在后台优化';
    } catch (error) {
      composerStatus.textContent = error.message || '参考图片添加失败';
      clearPendingReference();
    }
  });
  document.getElementById('newChatButton').addEventListener('click', newConversation);
  workspaceSwitch.addEventListener('click', () => switchWorkspace());
  mappingPlanButton.addEventListener('click', createMappingPlan);
  mappingStartButton.addEventListener('click', () => startMapping(false));
  mappingResumeButton.addEventListener('click', () => startMapping(true));
  mappingStopButton.addEventListener('click', stopMapping);
  mappingApplyButton.addEventListener('click', applyMappingResult);
  mappingLatestPreviewButton.addEventListener('click', () => {
    const url = mappingPreviewUrl(selectedPreviewName);
    if (url) openImage(url);
  });
  document.getElementById('menuButton').addEventListener('click', openSidebar);
  document.getElementById('mapButton').addEventListener('click', toggleMapPanel);
  document.getElementById('closeMapButton').addEventListener('click', closeMapPanel);
  scene3dInfoResizer.addEventListener('pointerdown', event => {
    event.preventDefault();
    const body = scene3dInfoResizer.parentElement;
    const updateWidth = clientX => {
      const rect = body.getBoundingClientRect();
      const maximum = Math.max(170, Math.min(420, rect.width * .48));
      const width = Math.max(150, Math.min(maximum, rect.right - clientX));
      body.style.setProperty('--scene-info-width', `${Math.round(width)}px`);
      scene3d?.resize?.();
      return Math.round(width);
    };
    let width = updateWidth(event.clientX);
    document.body.classList.add('resizing-scene-info');
    scene3dInfoResizer.setPointerCapture(event.pointerId);
    const move = moveEvent => { width = updateWidth(moveEvent.clientX); };
    const finish = () => {
      scene3dInfoResizer.removeEventListener('pointermove', move);
      scene3dInfoResizer.removeEventListener('pointerup', finish);
      scene3dInfoResizer.removeEventListener('pointercancel', finish);
      document.body.classList.remove('resizing-scene-info');
      localStorage.setItem('jaka-vision-scene-info-width', String(width));
    };
    scene3dInfoResizer.addEventListener('pointermove', move);
    scene3dInfoResizer.addEventListener('pointerup', finish);
    scene3dInfoResizer.addEventListener('pointercancel', finish);
  });
  reset3dViewButton.addEventListener('click', () => {
    if (!scene3d) return;
    reset3DFollowView();
    update3DCamera();
  });
  realView3dButton.addEventListener('click', () => {
    sync3DScene();
    if (!scene3d) return;
    scene3d.realView = !scene3d.realView;
    realView3dButton.setAttribute('aria-pressed', String(scene3d.realView));
    update3DVisibility();
  });
  trackPanelToggleButton.addEventListener('click', () => {
    const open = mapTrackPanel.classList.toggle('open');
    trackPanelToggleButton.classList.toggle('active', open);
    trackPanelToggleButton.setAttribute('aria-label', open ? '隐藏轨迹记录' : '显示轨迹记录');
  });
  capabilitySelect.addEventListener('change', applyCapabilityExample);
  planSelectedPatrol.addEventListener('click', () => sendMessage());
  document.getElementById('clearPatrolSelection').addEventListener('click', () => {
    if (sending) return;
    patrolSelectedIds = []; patrolMapSnapshot = null;
    refreshPatrolSetup(); renderMap();
  });
  backdrop.addEventListener('click', closeSidebar);
  document.getElementById('closeImageButton').addEventListener('click', () => imageDialog.close());
  imageDialog.addEventListener('click', event => { if (event.target === imageDialog) imageDialog.close(); });
  document.querySelectorAll('[data-prompt]').forEach(button => {
    button.addEventListener('click', () => sendMessage(button.dataset.prompt || ''));
  });
  currentTrackVisibleInput.addEventListener('change', () => {
    currentTrackVisible = currentTrackVisibleInput.checked;
    localStorage.setItem('jaka-vision-current-track-visible', String(currentTrackVisible));
    renderMap();
  });
  document.getElementById('saveTrackButton').addEventListener('click', async () => {
    const defaultName = `轨迹 ${new Date().toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}`;
    const name = prompt('给这条轨迹命名', defaultName);
    if (name === null) return;
    try {
      const result = await api('/api/tracks/save', { name: name.trim() || defaultName });
      savedTracks = result.tracks || [];
      if (result.track?.id) visibleTrackIds.add(String(result.track.id));
      storeVisibleTrackIds();
      if (robotData) robotData.track = [];
      renderTrackHistory();
      renderMap();
      composerStatus.textContent = '当前轨迹已保存';
    } catch (error) {
      composerStatus.textContent = `轨迹保存失败：${error.message}`;
    }
  });
  document.getElementById('clearCurrentTrackButton').addEventListener('click', async () => {
    if (!confirm('确定清空当前实时轨迹吗？已保存的历史轨迹不会受到影响。')) return;
    try {
      await api('/api/tracks/clear-current', {});
      if (robotData) robotData.track = [];
      renderMap();
      composerStatus.textContent = '当前实时轨迹已清空';
    } catch (error) {
      composerStatus.textContent = `轨迹清空失败：${error.message}`;
    }
  });
  document.getElementById('rotateLeftButton').addEventListener('click', () => {
    mapRotation = (mapRotation - 15) % 360;
    localStorage.setItem('jaka-vision-map-rotation', String(mapRotation));
    renderMap();
  });
  document.getElementById('rotateRightButton').addEventListener('click', () => {
    mapRotation = (mapRotation + 15) % 360;
    localStorage.setItem('jaka-vision-map-rotation', String(mapRotation));
    renderMap();
  });
  document.getElementById('resetMapButton').addEventListener('click', () => {
    mapZoom = DEFAULT_MAP_ZOOM; mapPanX = 0; mapPanY = 0; mapRotation = 0;
    localStorage.setItem('jaka-vision-map-rotation', '0');
    renderMap();
  });
  mapSelect.addEventListener('change', async () => {
    try {
      mapData = await api('/api/map/select', { name: mapSelect.value });
      selectedObjectId = null;
      welcomePickupAnnId = null;
      welcomeReturnAnnId = null;
      welcomeSelectionRole = capabilitySelect.value === 'welcome' ? 'pickup' : null;
      mapZoom = DEFAULT_MAP_ZOOM;
      mapPanX = 0;
      mapPanY = 0;
      renderMapLegend();
      renderObjectInspector();
      refreshWelcomeSetup();
      renderMap();
    } catch (error) {
      composerStatus.textContent = error.message;
    }
  });
  document.getElementById('uploadMapButton').addEventListener('click', () => mapFileInput.click());
  mapFileInput.addEventListener('change', async () => {
    const file = mapFileInput.files?.[0];
    if (!file) return;
    try {
      const graph = JSON.parse(await file.text());
      mapData = await api('/api/map/upload', { name: file.name, graph });
      const option = document.createElement('option');
      option.value = mapData.name;
      option.textContent = mapData.name;
      option.selected = true;
      mapSelect.append(option);
      selectedObjectId = null;
      welcomePickupAnnId = null;
      welcomeReturnAnnId = null;
      welcomeSelectionRole = capabilitySelect.value === 'welcome' ? 'pickup' : null;
      mapZoom = DEFAULT_MAP_ZOOM;
      mapPanX = 0;
      mapPanY = 0;
      renderMapLegend();
      renderObjectInspector();
      refreshWelcomeSetup();
      renderMap();
    } catch (error) {
      composerStatus.textContent = `地图加载失败：${error.message}`;
    } finally {
      mapFileInput.value = '';
    }
  });
  document.getElementById('uploadSlamButton').addEventListener('click', () => slamFileInput.click());
  slamFileInput.addEventListener('change', async () => {
    const file = slamFileInput.files?.[0];
    if (!file) return;
    try {
      if (file.size > 14 * 1024 * 1024) throw new Error('图片不能超过 14 MB');
      composerStatus.textContent = '正在加载 SLAM 图片';
      const imageBase64 = await fileAsBase64(file);
      slamData = await api('/api/slam/upload', { name: file.name, image_base64: imageBase64 });
      showSlamCalibration(slamData);
      slamVisible = true;
      slamVisibleInput.checked = true;
      localStorage.setItem('jaka-vision-slam-visible', 'true');
      mapZoom = DEFAULT_MAP_ZOOM; mapPanX = 0; mapPanY = 0;
      renderMap();
      composerStatus.textContent = 'SLAM 图片已加载，请校准位置';
      slamCalibrationPanel.hidden = false;
    } catch (error) {
      composerStatus.textContent = `SLAM 图片加载失败：${error.message}`;
    } finally {
      slamFileInput.value = '';
    }
  });
  slamVisibleInput.addEventListener('change', () => {
    slamVisible = slamVisibleInput.checked;
    localStorage.setItem('jaka-vision-slam-visible', String(slamVisible));
    renderMap();
  });
  document.getElementById('refreshSlamButton').addEventListener('click', async event => {
    const button = event.currentTarget;
    button.disabled = true;
    composerStatus.textContent = '正在重新读取本地 slam.png';
    try {
      slamData = await api('/api/slam/reload-local', {});
      showSlamCalibration(slamData);
      mapZoom = DEFAULT_MAP_ZOOM; mapPanX = 0; mapPanY = 0;
      renderMap();
      composerStatus.textContent = '本地 SLAM 底图已重新加载';
    } catch (error) {
      composerStatus.textContent = `本地底图加载失败：${error.message}`;
    } finally {
      button.disabled = false;
    }
  });
  document.getElementById('slamCalibrationButton').addEventListener('click', () => {
    slamCalibrationPanel.hidden = !slamCalibrationPanel.hidden;
  });
  const previewSlamCalibration = () => {
    const value = readSlamCalibration();
    if (!slamData?.available || !Object.values(value).every(Number.isFinite) || value.resolution <= 0) return;
    slamData.calibration = value;
    renderMap();
  };
  for (const control of [slamOriginX, slamOriginY, slamResolution, slamRotation, slamOpacity]) {
    control.addEventListener('input', previewSlamCalibration);
  }
  document.getElementById('autoSlamCalibrationButton').addEventListener('click', async () => {
    try {
      slamData = await api('/api/slam/auto-calibrate', {});
      showSlamCalibration(slamData);
      mapZoom = DEFAULT_MAP_ZOOM; mapPanX = 0; mapPanY = 0;
      renderMap();
      composerStatus.textContent = '已按采集轨迹自动对齐 SLAM 底图';
    } catch (error) {
      composerStatus.textContent = `SLAM 自动对齐失败：${error.message}`;
    }
  });
  document.getElementById('saveSlamCalibrationButton').addEventListener('click', async () => {
    try {
      slamData = await api('/api/slam/calibration', { calibration: readSlamCalibration() });
      showSlamCalibration(slamData);
      renderMap();
      composerStatus.textContent = 'SLAM 标定已保存';
    } catch (error) {
      composerStatus.textContent = `SLAM 标定失败：${error.message}`;
    }
  });
  document.getElementById('goObjectButton').addEventListener('click', () => {
    const object = selectedMapObject();
    if (!object) return;
    const name = object.category_zh || object.category || `物体${object.ann_id}`;
    openMapPanel();
    sendMessage(`前往${name}（ann_id=${object.ann_id}）附近`);
  });
  document.getElementById('observeObjectButton').addEventListener('click', () => {
    const object = selectedMapObject();
    if (!object) return;
    const name = object.category_zh || object.category || `物体${object.ann_id}`;
    openMapPanel();
    sendMessage(`前往${name}（ann_id=${object.ann_id}）附近观察，并告诉我现场情况`);
  });

  mapStage.addEventListener('wheel', event => {
    event.preventDefault();
    mapZoom = Math.max(.45, Math.min(8, mapZoom * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
    renderMap();
  }, { passive: false });
  mapStage.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    event.preventDefault();
    window.getSelection?.()?.removeAllRanges();
    mapDrag = { x: event.clientX, y: event.clientY, panX: mapPanX, panY: mapPanY, moved: false, pointerId: event.pointerId };
    mapStage.classList.add('dragging');
  });
  mapStage.addEventListener('pointermove', event => {
    if (!mapDrag) return;
    const dx = event.clientX - mapDrag.x;
    const dy = event.clientY - mapDrag.y;
    if (!mapDrag.moved && Math.abs(dx) + Math.abs(dy) > 4) {
      mapDrag.moved = true;
      mapStage.setPointerCapture?.(event.pointerId);
    }
    mapPanX = mapDrag.panX - dx / Math.max(1, mapStage.clientWidth) * mapVisibleWorld.x;
    mapPanY = mapDrag.panY + dy / Math.max(1, mapStage.clientHeight) * mapVisibleWorld.y;
    renderMap();
  });
  mapStage.addEventListener('pointerup', event => {
    if (!mapDrag) return;
    suppressMapClick = mapDrag.moved;
    mapDrag = null;
    mapStage.classList.remove('dragging');
    if (mapStage.hasPointerCapture?.(event.pointerId)) mapStage.releasePointerCapture(event.pointerId);
    if (suppressMapClick) setTimeout(() => { suppressMapClick = false; }, 80);
  });
  mapStage.addEventListener('pointercancel', () => {
    mapDrag = null;
    mapStage.classList.remove('dragging');
  });
  mapStage.addEventListener('selectstart', event => event.preventDefault());
  mapStage.addEventListener('dblclick', () => { mapZoom = DEFAULT_MAP_ZOOM; mapPanX = 0; mapPanY = 0; renderMap(); });

  mapResizer.addEventListener('pointerdown', event => {
    if (window.innerWidth <= 1180) return;
    const startX = event.clientX;
    const startWidth = mapPanel.getBoundingClientRect().width;
    const move = moveEvent => {
      const minimumMain = 520;
      const minWidth = 320;
      const maxWidth = Math.max(minWidth, window.innerWidth - 264 - minimumMain);
      const nextWidth = Math.max(minWidth, Math.min(maxWidth, startWidth + startX - moveEvent.clientX));
      document.documentElement.style.setProperty('--map-width', `${nextWidth}px`);
      renderMap();
    };
    const stop = () => {
      document.body.classList.remove('resizing-map');
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      const width = Math.round(mapPanel.getBoundingClientRect().width);
      localStorage.setItem('jaka-vision-map-width', String(width));
    };
    document.body.classList.add('resizing-map');
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
    event.preventDefault();
  });

  ensureConversation();
  renderAll();
  switchWorkspace(workspaceMode).catch(error => { mappingHeaderStatus.textContent = error.message; });
  autoResize();
  checkHealth();
  setInterval(checkHealth, 15000);
  loadMapCatalog().catch(error => { mapLoading.textContent = `地图加载失败：${error.message}`; });
  loadSlamMap().catch(error => { slamName.textContent = `SLAM 加载失败：${error.message}`; });
  loadTrackHistory().catch(error => {
    trackHistoryEmpty.textContent = `轨迹读取失败：${error.message}`;
  });
  pollRobotStatus();
  pollMappingStatus();
  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('/service-worker.js').catch(error => {
      console.warn('PWA Service Worker 注册失败', error);
    });
  }
})();
