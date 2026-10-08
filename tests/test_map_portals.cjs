const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const html=require('./helpers/frontend_source.cjs').combined;
const match=/  function standaloneDoorways\(data\) \{[\s\S]*?\n  }/.exec(html);
assert.ok(match);
const c=vm.createContext({}); vm.runInContext(match[0],c);
const data={objects:[{ann_id:1}],doorways:[
 {portal_id:'linked',anchor_ann_id:1,floor_xy:[0,0]},
 {portal_id:'glass',anchor_ann_id:null,member_ids:[],floor_xy:[3,10]},
 {portal_id:'invalid',floor_xy:[null,2]},
 {portal_id:'member',member_ids:[1],floor_xy:[3,3]}]};
assert.deepEqual(Array.from(c.standaloneDoorways(data),p=>p.portal_id),['glass']);
assert.equal(c.standaloneDoorways({objects:[]}).length,0);
assert.equal(c.standaloneDoorways({objects:[{ann_id:1000001}],doorways:[
  {portal_id:'glass',navigation_ann_id:1000001,floor_xy:[3,10]}]}).length,0);
const portalRender = /for \(const portal of standaloneDoorways\(mapData\)\) \{[\s\S]*?sceneLayer.append\(group\);/.exec(html);
assert.ok(portalRender);
assert.ok(portalRender[0].includes("appendObjectIcon(group, 'door'"));
assert.ok(!portalRender[0].includes("svgNode('text'"), 'Portal markers must be icon-only');
assert.ok(portalRender[0].includes("'aria-label'"), 'Keep the accessible marker name');
for(const m of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(m[1]);
console.log('Portal layer and script syntax tests passed');
