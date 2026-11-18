import {cpSync, mkdirSync, rmSync} from 'node:fs';
rmSync('dist', {recursive:true, force:true});
mkdirSync('dist');
cpSync('web', 'dist', {recursive:true});
console.log('Built Power Guard static website.');
