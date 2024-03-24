import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { SeiteAboutUsComponent } from '../seite-about-us/seite-about-us.component';
import { SeiteWelcomeComponent } from '../seite-welcome/seite-welcome.component';
import { SeiteAboutThisComponent } from '../seite-about-this/seite-about-this.component';
import { SeiteTechnologyComponent } from '../seite-technology/seite-technology.component';
import { SeiteDatabaseComponent } from '../seite-database/seite-database.component';
import { SeiteNnComponent } from '../seite-nn/seite-nn.component';
import { SeiteStartComponent } from '../seite-start/seite-start.component';
import { SeiteMainExactorComponent } from '../seite-main-exactor/seite-main-exactor.component'
import { SeiteGuideComponent } from '../seite-guide/seite-guide.component';
import { SeiteFeedbackComponent } from '../seite-feedback/seite-feedback.component';


const routes: Routes = [
  { path: '', component: SeiteWelcomeComponent },
  { path: 'about-us', component: SeiteAboutUsComponent },
  { path: 'about-this', component: SeiteAboutThisComponent },
  { path: 'technology', component: SeiteTechnologyComponent },
  { path: 'database', component: SeiteDatabaseComponent },
  { path: 'nn', component: SeiteNnComponent },
  { path: 'start', component: SeiteStartComponent, children: [
    { path: '', redirectTo: 'guide', pathMatch: 'full' }, // Redirect /nn to /nn/guide
    { path: 'guide', component: SeiteGuideComponent },
    { path: 'feedback', component: SeiteFeedbackComponent },
    { path: 'main-exactor', component: SeiteMainExactorComponent },
  ]},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
