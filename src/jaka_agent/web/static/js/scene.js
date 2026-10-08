    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const canvas = document.getElementById('sceneCanvas');
    const stage = document.getElementById('stage');
    const loading = document.getElementById('loading');
    const main = document.getElementById('main');
    const info = document.getElementById('info');
    const objectSelect = document.getElementById('objectSelect');
    const sourceName = document.getElementById('sourceName');
    const cameraState = document.getElementById('cameraState');
    const objectCount = document.getElementById('objectCount');
    const legend = document.getElementById('legend');

    const categoryPalette = ['#4f8fc9', '#43a267', '#e59b38', '#d56b6b', '#7a70c9', '#45a99a', '#ba6b9f', '#788792'];
    const categoryNames = {
      'shrub or bush': '灌木', 'ordinary office door': '普通办公室门', 'blue chair': '蓝色椅子',
      'black flight case': '黑色航空箱', 'window': '窗户', 'white utility table': '白色工作台',
      'wooden storage cabinet': '木质储物柜', 'leather armchair': '皮质扶手椅',
      'potted plant': '盆栽', 'glass table': '玻璃茶几', 'silver elevator door': '银色电梯门',
      'guest pickup point': '接客点', 'guest drop-off point': '送客点'
    };
    const infoFields = [
      ['颜色', 'color'], ['材质', 'material'], ['尺寸', 'size'], ['状态', 'state'],
      ['位置', 'position'], ['朝向', 'direction'], ['风格', 'style'], ['用途', 'func_desc']
    ];

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = false;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f1719');
    scene.fog = new THREE.FogExp2('#0f1719', 0.0105);

    const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.085;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.4;
    controls.maxDistance = 500;
    controls.minPolarAngle = 0.03;
    controls.maxPolarAngle = Math.PI / 2 - 0.015;
    controls.zoomToCursor = true;

    scene.add(new THREE.HemisphereLight('#dfeff0', '#263536', 1.45));
    const keyLight = new THREE.DirectionalLight('#ffffff', 1.15);
    keyLight.position.set(18, 32, 12);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight('#65aeb4', 0.35);
    fillLight.position.set(-20, 15, -15);
    scene.add(fillLight);

    const world = new THREE.Group();
    const relationWorld = new THREE.Group();
    const robotWorld = new THREE.Group();
    scene.add(world, relationWorld, robotWorld);
    let grid = null;
    let ground = null;
    let graph = null;
    let selectedId = null;
    let sceneBounds = new THREE.Box3();
    let labelsVisible = true;
    let relationsVisible = false;
    let gridVisible = true;
    let robotVisible = true;
    const objectItems = new Map();
    const pickMeshes = [];
    const relationItems = [];
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function categoryName(object) {
      const raw = String(object.category || '').trim();
      return String(object.category_zh || categoryNames[raw.toLowerCase()] || raw || '物体');
    }

    function categoryColor(category) {
      const value = String(category || '').toLowerCase();
      if (value.includes('接客点') || value.includes('pickup')) return '#d97706';
      if (value.includes('送客点') || value.includes('drop-off')) return '#2563eb';
      if (value.includes('盆栽') || value.includes('植物') || value.includes('plant')) return '#3f8a5c';
      if (value.includes('灌木') || value.includes('shrub') || value.includes('bush')) return '#4f8b67';
      if (value.includes('电梯') || value.includes('elevator')) return '#2f8f83';
      if (value.includes('门') || value.includes('door')) return '#5579b8';
      if (value.includes('窗') || value.includes('window')) return '#358aa8';
      if (value.includes('储物柜') || value.includes('cabinet')) return '#4e9a68';
      if (value.includes('椅') || value.includes('chair') || value.includes('armchair')) return '#b65d83';
      if (value.includes('航空箱') || value.includes('箱') || value.includes('case')) return '#3979b7';
      if (value.includes('茶几') || value.includes('glass table') || value.includes('coffee table')) return '#b56b9a';
      if (value.includes('工作台') || value.includes('操作台') || value.includes('桌') || value.includes('table') || value.includes('desk')) return '#7465ad';
      let hash = 0;
      for (const char of String(category || 'Unknown')) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
      return categoryPalette[hash % categoryPalette.length];
    }

    function normalizeGraph(data, name = 'scene_graph.json') {
      const rawObjects = Array.isArray(data?.objects) ? data.objects : Object.values(data?.objects || {});
      const objects = [];
      for (const source of rawObjects) {
        if (!source || typeof source !== 'object') continue;
        const geometry = source.geometry || {};
        const box = source.box3d || {};
        const center = box.center || geometry.center_world || source.floor_xy;
        const explicitSize = box.size;
        const half = geometry.aabb_half_sizes_world;
        const size = explicitSize || (Array.isArray(half) ? half.map(value => Number(value) * 2) : null);
        if (!Array.isArray(center) || center.length < 2 || !Array.isArray(size) || size.length < 2) continue;
        const normalizedCenter = [Number(center[0]), Number(center[1]), Number(center[2] ?? Number(size[2] || 1) / 2)];
        const normalizedSize = [Math.max(.08, Number(size[0]) || .6), Math.max(.08, Number(size[1]) || .6), Math.max(.08, Number(size[2]) || 1)];
        if (!normalizedCenter.every(Number.isFinite) || !normalizedSize.every(Number.isFinite)) continue;
        objects.push({
          ...source,
          ann_id: source.ann_id ?? objects.length + 1,
          category_zh: source.category_zh || categoryNames[String(source.category || '').toLowerCase()] || source.category || '物体',
          floor_xy: source.floor_xy || normalizedCenter.slice(0, 2),
          box3d: { ...box, center: normalizedCenter, size: normalizedSize }
        });
      }
      const relationGroups = data?.relationships || {};
      const pos = Array.isArray(data?.pos_relationships) ? data.pos_relationships : (relationGroups.positional || []);
      const func = Array.isArray(data?.func_relationships) ? data.func_relationships : (relationGroups.functional || []);
      return { name: data?.name || name, objects, pos_relationships: pos || [], func_relationships: func || [] };
    }

    function clearGroup(group) {
      while (group.children.length) {
        const child = group.children.pop();
        child.traverse(node => {
          node.geometry?.dispose?.();
          if (node.material?.map) node.material.map.dispose?.();
          if (Array.isArray(node.material)) node.material.forEach(material => material.dispose?.());
          else node.material?.dispose?.();
        });
      }
    }

    function makeLabel(text, color) {
      const surface = document.createElement('canvas');
      const context = surface.getContext('2d');
      context.font = '600 16px "Microsoft YaHei", sans-serif';
      const width = Math.min(260, Math.max(58, Math.ceil(context.measureText(text).width + 22)));
      surface.width = width;
      surface.height = 34;
      context.fillStyle = 'rgba(8,19,21,.82)';
      context.beginPath();
      context.roundRect(1, 1, width - 2, 32, 6);
      context.fill();
      context.strokeStyle = color;
      context.globalAlpha = .82;
      context.lineWidth = 1.5;
      context.stroke();
      context.globalAlpha = 1;
      context.fillStyle = '#e7f2f2';
      context.font = '600 16px "Microsoft YaHei", sans-serif';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(text, width / 2, 17, width - 12);
      const texture = new THREE.CanvasTexture(surface);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.minFilter = THREE.LinearFilter;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, opacity: .92 }));
      sprite.scale.set(width / 215, .16, 1);
      sprite.userData.baseScale = sprite.scale.clone();
      sprite.renderOrder = 20;
      return sprite;
    }

    function endpointId(relation, side) {
      const nested = relation?.[`${side}_obj`] || relation?.[side] || {};
      const direct = relation?.[`${side}_ann_id`] ?? relation?.[`${side}_id`];
      return String(nested?.ann_id ?? nested?.id ?? direct ?? '');
    }

    function relationName(relation) {
      return String(relation?.relation || relation?.type || relation?.name || (relation?.relation_type === 'functional' ? '功能关联' : '空间关联'));
    }

    function sceneRelations() {
      if (!graph) return [];
      return [
        ...(graph.pos_relationships || []).map(value => ({ ...value, relation_type: 'positional' })),
        ...(graph.func_relationships || []).map(value => ({ ...value, relation_type: 'functional' }))
      ];
    }

    function buildScene(nextGraph, sourceLabel) {
      graph = nextGraph;
      selectedId = null;
      objectItems.clear();
      pickMeshes.length = 0;
      relationItems.length = 0;
      clearGroup(world);
      clearGroup(relationWorld);
      clearGroup(robotWorld);
      if (grid) { scene.remove(grid); grid.geometry.dispose(); grid.material.dispose(); }
      if (ground) { scene.remove(ground); ground.geometry.dispose(); ground.material.dispose(); }

      sceneBounds = new THREE.Box3();
      const categoryCounts = new Map();
      for (const object of graph.objects) {
        const center = object.box3d.center;
        const size = object.box3d.size;
        const name = categoryName(object);
        const color = categoryColor(name);
        const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
        const material = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: .24, depthWrite: false, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.annId = String(object.ann_id);
        const edgeMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: .68 });
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), edgeMaterial);
        const group = new THREE.Group();
        group.position.set(center[0], center[2], -center[1]);
        group.add(mesh, edges);
        const label = makeLabel(name, color);
        label.position.set(0, size[2] / 2 + .18, 0);
        group.add(label);
        world.add(group);
        const halfSize = new THREE.Vector3(size[0] / 2, size[2] / 2, size[1] / 2);
        sceneBounds.expandByPoint(group.position.clone().sub(halfSize));
        sceneBounds.expandByPoint(group.position.clone().add(halfSize));
        const item = { object, group, mesh, edges, label, material, edgeMaterial, color, name };
        objectItems.set(String(object.ann_id), item);
        pickMeshes.push(mesh);
        categoryCounts.set(name, (categoryCounts.get(name) || 0) + 1);
      }

      for (const relation of sceneRelations()) {
        const head = objectItems.get(endpointId(relation, 'head'));
        const tail = objectItems.get(endpointId(relation, 'tail'));
        if (!head || !tail || head === tail) continue;
        const color = relation.relation_type === 'functional' ? '#d99a35' : '#638d91';
        const points = [head.group.position.clone(), tail.group.position.clone()];
        points[0].y += .08;
        points[1].y += .08;
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: .72 })
        );
        line.visible = relationsVisible;
        relationWorld.add(line);
        relationItems.push({ relation, line, headId: String(head.object.ann_id), tailId: String(tail.object.ann_id) });
      }

      const size = sceneBounds.getSize(new THREE.Vector3());
      const center = sceneBounds.getCenter(new THREE.Vector3());
      const extent = Math.max(size.x, size.z, 24);
      ground = new THREE.Mesh(
        new THREE.PlaneGeometry(extent * 1.55, extent * 1.55),
        new THREE.MeshLambertMaterial({ color: '#162123', transparent: true, opacity: .62 })
      );
      ground.rotation.x = -Math.PI / 2;
      ground.position.set(center.x, -.04, center.z);
      scene.add(ground);
      grid = new THREE.GridHelper(extent * 1.5, Math.max(20, Math.round(extent / 2)), '#355154', '#243638');
      grid.position.set(center.x, 0, center.z);
      grid.material.transparent = true;
      grid.material.opacity = .28;
      grid.visible = gridVisible;
      scene.add(grid);

      rebuildLegend(categoryCounts);
      rebuildObjectSelect();
      sourceName.textContent = `${sourceLabel || graph.name} · ${graph.objects.length} 个拟物体 · ${relationItems.length} 条空间关系`;
      objectCount.textContent = `${graph.objects.length} 个三维物体框`;
      renderInfo(null);
      fitScene('overview');
      loadRobotState();
      loading.hidden = true;
    }

    function rebuildLegend(counts) {
      legend.querySelectorAll('.legend-item').forEach(node => node.remove());
      const spacer = legend.querySelector('.footer-spacer');
      const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
      for (const [name, count] of entries) {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<i style="background:${categoryColor(name)}"></i>${name} ×${count}`;
        legend.insertBefore(item, spacer);
      }
    }

    function rebuildObjectSelect() {
      objectSelect.replaceChildren(new Option('选择拟物体以聚焦', ''));
      const objects = [...graph.objects].sort((a, b) => Number(a.ann_id) - Number(b.ann_id));
      for (const object of objects) objectSelect.append(new Option(`${categoryName(object)} · #${object.ann_id}`, String(object.ann_id)));
    }

    function setButtonState(id, value) {
      document.getElementById(id).setAttribute('aria-pressed', String(value));
    }

    function fitScene(preset = 'overview') {
      if (sceneBounds.isEmpty()) return;
      const center = sceneBounds.getCenter(new THREE.Vector3());
      const size = sceneBounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.length() * .5, 4);
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * .5)) * 1.05;
      const directions = {
        overview: new THREE.Vector3(1, .72, 1),
        top: new THREE.Vector3(0, 1, .0001),
        front: new THREE.Vector3(0, .18, 1),
        side: new THREE.Vector3(1, .18, 0)
      };
      const direction = (directions[preset] || directions.overview).normalize();
      camera.position.copy(center).add(direction.multiplyScalar(distance));
      controls.target.copy(center);
      controls.maxDistance = Math.max(distance * 5, 100);
      camera.near = Math.max(.03, distance / 1000);
      camera.far = Math.max(600, distance * 12);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function focusObject(id) {
      const item = objectItems.get(String(id));
      if (!item) return;
      selectObject(id);
      const size = item.object.box3d.size;
      const radius = Math.max(...size, .8);
      const direction = camera.position.clone().sub(controls.target).normalize();
      controls.target.copy(item.group.position);
      camera.position.copy(item.group.position).add(direction.multiplyScalar(Math.max(radius * 5.2, 3.2)));
      controls.update();
    }

    function selectObject(id) {
      selectedId = id == null ? null : String(id);
      objectSelect.value = selectedId || '';
      for (const [itemId, item] of objectItems) {
        const selected = itemId === selectedId;
        item.material.opacity = selected ? .48 : .24;
        item.edgeMaterial.color.set(selected ? '#38f0f4' : item.color);
        item.edgeMaterial.opacity = selected ? 1 : .74;
        item.label.scale.copy(item.label.userData.baseScale).multiplyScalar(selected ? 1.12 : 1);
      }
      renderInfo(selectedId ? objectItems.get(selectedId)?.object : null);
    }

    function relatedToObject(id) {
      const rows = [];
      for (const item of relationItems) {
        if (item.headId !== String(id) && item.tailId !== String(id)) continue;
        const otherId = item.headId === String(id) ? item.tailId : item.headId;
        const other = objectItems.get(otherId);
        if (other) rows.push(`${relationName(item.relation)} · ${other.name} #${other.object.ann_id}`);
      }
      return rows;
    }

    function renderInfo(object) {
      document.getElementById('infoEmpty').hidden = Boolean(object);
      const content = document.getElementById('infoContent');
      content.hidden = !object;
      if (!object) return;
      const name = categoryName(object);
      const box = object.box3d;
      document.getElementById('infoName').textContent = name;
      document.getElementById('infoMeta').textContent = `#${object.ann_id} · (${box.center.map(value => Number(value).toFixed(2)).join(', ')})`;
      const list = document.getElementById('propertyList');
      list.replaceChildren();
      const values = [
        ['类别', object.category || '—'],
        ['三维尺寸', box.size.map(value => Number(value).toFixed(2)).join(' × ')],
        ...infoFields.map(([label, key]) => [label, object[key]]).filter(([, value]) => String(value || '').trim())
      ];
      for (const [label, value] of values) {
        const row = document.createElement('div');
        row.className = 'property-row';
        const dt = document.createElement('dt');
        dt.textContent = label;
        const dd = document.createElement('dd');
        if (label === '颜色') {
          dd.innerHTML = `<i class="swatch" style="background:${categoryColor(name)}"></i>${String(value)}`;
        } else dd.textContent = String(value);
        row.append(dt, dd);
        list.append(row);
      }
      const relationList = document.getElementById('relationList');
      relationList.replaceChildren();
      const rows = relatedToObject(object.ann_id);
      for (const text of rows.length ? rows : ['暂无已知关系']) {
        const row = document.createElement('div');
        row.className = 'relation';
        row.textContent = text;
        relationList.append(row);
      }
    }

    function pickObject(event, focus = false) {
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(pickMeshes, false)[0];
      if (!hit) return;
      const id = hit.object.userData.annId;
      if (focus) focusObject(id); else selectObject(id);
    }

    async function loadRobotState() {
      clearGroup(robotWorld);
      try {
        const response = await fetch('/api/robot/status', { cache: 'no-store' });
        if (!response.ok) return;
        const data = await response.json();
        const robot = data.robot || {};
        const pose = robot.pose;
        if (!pose) return;
        const material = new THREE.MeshPhongMaterial({ color: '#e8ffff', emissive: '#71cfd3', emissiveIntensity: .22 });
        const marker = new THREE.Mesh(new THREE.SphereGeometry(.26, 22, 16), material);
        marker.position.set(Number(pose.x), 1.35, -Number(pose.y));
        robotWorld.add(marker);
        const theta = Number(pose.theta || 0);
        const forward = new THREE.Vector3(Math.cos(theta), 0, -Math.sin(theta));
        const right = new THREE.Vector3(-forward.z, 0, forward.x);
        const origin = marker.position.clone();
        const length = 2.2;
        const tip = origin.clone().add(forward.clone().multiplyScalar(length));
        const leftTip = tip.clone().add(right.clone().multiplyScalar(-.9));
        const rightTip = tip.clone().add(right.clone().multiplyScalar(.9));
        const frustum = new THREE.LineSegments(
          new THREE.BufferGeometry().setFromPoints([origin, leftTip, origin, rightTip, leftTip, rightTip]),
          new THREE.LineBasicMaterial({ color: '#2bd0d5', transparent: true, opacity: .92 })
        );
        robotWorld.add(frustum);
        const track = Array.isArray(robot.track) ? robot.track : [];
        if (track.length > 1) {
          const points = track.map(value => new THREE.Vector3(Number(value[0]), .025, -Number(value[1])));
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(points),
            new THREE.LineBasicMaterial({ color: '#42b9a7', transparent: true, opacity: .85 })
          );
          robotWorld.add(line);
        }
        robotWorld.visible = robotVisible;
      } catch (_) {
        // 独立打开 HTML 或服务暂不可用时，不影响场景图浏览。
      }
    }

    function resize() {
      const rect = stage.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / rect.height;
      camera.updateProjectionMatrix();
    }

    function updateCameraState() {
      cameraState.textContent = `camera (${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${(-camera.position.z).toFixed(2)})  target (${controls.target.x.toFixed(2)}, ${controls.target.y.toFixed(2)}, ${(-controls.target.z).toFixed(2)})`;
    }

    function saveImage() {
      renderer.render(scene, camera);
      canvas.toBlob(blob => {
        if (!blob) return;
        const link = document.createElement('a');
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `拟物体三维地图_${stamp}.png`;
        link.href = URL.createObjectURL(blob);
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      }, 'image/png');
    }

    async function loadDefaultGraph() {
      loading.hidden = false;
      try {
        const response = await fetch('/api/map', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const normalized = normalizeGraph(data, data.name || 'zmq_scene_graph.json');
        if (!normalized.objects.length) throw new Error('场景图中没有有效三维物体');
        buildScene(normalized, data.name || 'zmq_scene_graph.json');
      } catch (error) {
        loading.textContent = `自动加载失败：${error.message}。请点击“载入 JSON”选择本地场景图。`;
      }
    }

    document.getElementById('overviewButton').addEventListener('click', () => fitScene('overview'));
    document.getElementById('topButton').addEventListener('click', () => fitScene('top'));
    document.getElementById('frontButton').addEventListener('click', () => fitScene('front'));
    document.getElementById('sideButton').addEventListener('click', () => fitScene('side'));
    document.getElementById('focusButton').addEventListener('click', () => selectedId && focusObject(selectedId));
    objectSelect.addEventListener('change', () => objectSelect.value && focusObject(objectSelect.value));
    document.getElementById('labelsButton').addEventListener('click', event => {
      labelsVisible = !labelsVisible;
      for (const item of objectItems.values()) item.label.visible = labelsVisible;
      event.currentTarget.setAttribute('aria-pressed', String(labelsVisible));
    });
    document.getElementById('relationsButton').addEventListener('click', event => {
      relationsVisible = !relationsVisible;
      relationWorld.visible = relationsVisible;
      event.currentTarget.setAttribute('aria-pressed', String(relationsVisible));
    });
    document.getElementById('gridButton').addEventListener('click', event => {
      gridVisible = !gridVisible;
      if (grid) grid.visible = gridVisible;
      event.currentTarget.setAttribute('aria-pressed', String(gridVisible));
    });
    document.getElementById('robotButton').addEventListener('click', event => {
      robotVisible = !robotVisible;
      robotWorld.visible = robotVisible;
      event.currentTarget.setAttribute('aria-pressed', String(robotVisible));
    });
    document.getElementById('infoButton').addEventListener('click', event => {
      const visible = !main.classList.toggle('info-collapsed');
      info.hidden = !visible;
      event.currentTarget.setAttribute('aria-pressed', String(visible));
      setTimeout(resize, 180);
    });
    document.getElementById('loadButton').addEventListener('click', () => document.getElementById('jsonFile').click());
    document.getElementById('saveButton').addEventListener('click', saveImage);
    document.getElementById('jsonFile').addEventListener('change', async event => {
      const file = event.target.files?.[0];
      if (!file) return;
      loading.hidden = false;
      loading.textContent = `正在读取 ${file.name}…`;
      try {
        const data = JSON.parse(await file.text());
        const normalized = normalizeGraph(data, file.name);
        if (!normalized.objects.length) throw new Error('文件中没有有效三维物体');
        buildScene(normalized, file.name);
      } catch (error) {
        loading.hidden = false;
        loading.textContent = `载入失败：${error.message}`;
      } finally {
        event.target.value = '';
      }
    });
    canvas.addEventListener('click', event => pickObject(event, false));
    canvas.addEventListener('dblclick', event => pickObject(event, true));
    controls.addEventListener('change', updateCameraState);
    new ResizeObserver(resize).observe(stage);

    function animate() {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }
    resize();
    updateCameraState();
    requestAnimationFrame(animate);
    loadDefaultGraph();
