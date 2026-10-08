const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'..','robot_web_page.html'),'utf8');
function source(name){
  const start=new RegExp(`^  (?:async )?function ${name}\\(`,'m').exec(html).index;
  const rest=html.slice(start),end=/\n  }\r?\n/.exec(rest);
  return rest.slice(0,end.index+end[0].length);
}
function node(){return {children:[],hidden:false,value:'-1',replaceChildren(){this.children=[]},append(x){this.children.push(x)},
  addEventListener(name,fn){this[name]=fn}}}
const context=vm.createContext({
  patrolSelectedIds:[],patrolMapSnapshot:null,mapData:{snapshot:'v1',objects:[1,2,3].map(id=>({ann_id:id,category_zh:'地点'+id,floor_xy:[id,0]}))},
  patrolSetup:node(),patrolSelectedList:node(),patrolSelectionCount:node(),patrolSelectionHint:node(),patrolRounds:node(),planSelectedPatrol:node(),
  objectInspector:{querySelector:()=>node()},document:{createElement:node},capabilitySelect:{value:'patrol'},sending:false,
  conversationStoreReady:true,backendOnline:true,speechState:'idle',renderMap(){},
});
for(const name of ['refreshPatrolSetup','togglePatrolPoint','selectedPatrolRequest'])vm.runInContext(source(name),context);
const objects=context.mapData.objects;
context.refreshPatrolSetup();assert.equal(context.planSelectedPatrol.disabled,true);
context.togglePatrolPoint(objects[2]);context.togglePatrolPoint(objects[0]);context.togglePatrolPoint(objects[1]);
assert.equal(JSON.stringify(context.patrolSelectedIds),'[3,1,2]');
assert.equal(context.planSelectedPatrol.disabled,false);
let request=context.selectedPatrolRequest();assert.equal(JSON.stringify(request.target_ann_ids),'[3,1,2]');
assert.equal(request.rounds,-1);assert.equal(request.map_snapshot,'v1');
context.togglePatrolPoint(objects[0]);assert.equal(JSON.stringify(context.patrolSelectedIds),'[3,2]');
context.patrolRounds.value='2';assert.equal(context.selectedPatrolRequest().rounds,2);
context.patrolSelectedList.children[0].click();assert.equal(JSON.stringify(context.patrolSelectedIds),'[2]');
assert.throws(()=>context.selectedPatrolRequest(),/至少/);
context.togglePatrolPoint({ann_id:'portal',floor_xy:[1,2]});assert.equal(context.patrolSelectedIds.length,1);
context.mapData.snapshot='v2';context.refreshPatrolSetup();assert.equal(context.patrolSelectedIds.length,0);
context.sending=true;context.togglePatrolPoint(objects[0]);assert.equal(context.patrolSelectedIds.length,0);
assert.ok(html.includes("api('/api/task/patrol/plan'"));
console.log('Patrol selection UI: exact order, toggle/removal, rounds, stale map, invalid portal and in-flight guard passed');
