const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'robot_web_page.html'), 'utf8');
const start = html.indexOf('      const pending = task.guest_confirmation;');
const end = html.indexOf("    if (task.kind === 'find_object')", start);
const block = html.slice(start, end).replace(/\s*}\s*$/, '');
function element(tag) {
  return {tag, children: [], events: {}, style: {}, append(...items) {this.children.push(...items)},
          addEventListener(name, fn) {this.events[name] = fn}};
}
function render(status='running', canceled=false) {
  const tool = element('section');
  const calls = [];
  const context = vm.createContext({tool, task: {id:'A', status, cancel_requested:canceled,
    guest_confirmation:{id:'photo-1',status:'pending',image_url:'/captures/photo.jpg'}},
    document:{createElement:element}, navigator:{onLine:true}, robotData:{},
    executionBlocker:()=>'', openImage(){}, confirmWelcomeGuest:async(...args)=>calls.push(args)});
  vm.runInContext(block, context);
  return {tool,calls};
}
(async () => {
  const {tool,calls} = render();
  const [photo,note,buttons] = tool.children[0].children;
  const [reject,accept] = buttons.children;
  assert.equal(photo.src, '/captures/photo.jpg');
  assert.equal(accept.disabled,true);
  photo.events.load();
  assert.equal(accept.disabled,false);
  await accept.events.click();
  assert.equal(calls[0][1],'photo-1');
  assert.equal(calls[0][2],true);
  assert.equal(accept.disabled,true);
  const rejected=render();
  await rejected.tool.children[0].children[2].children[0].events.click();
  assert.equal(rejected.calls[0][2],false);
  const broken=render();
  const [brokenPhoto,brokenNote,brokenButtons]=broken.tool.children[0].children;
  brokenPhoto.events.error();
  assert.equal(brokenButtons.children[1].disabled,true);
  assert.match(brokenNote.textContent,/加载失败/);
  for (const state of ['canceled','succeeded','canceling']) assert.equal(render(state).tool.children.length,0);
  assert.equal(render('running',true).tool.children.length,0);
  console.log('Welcome UI: photo gating, confirmation, rejection, cancellation passed');
})().catch(e=>{console.error(e);process.exitCode=1});
