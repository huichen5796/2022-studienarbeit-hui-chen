import { Component } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { filter, Observable } from 'rxjs';

@Component({
  selector: 'app-seite-start',
  templateUrl: './seite-start.component.html',
  styleUrls: ['./seite-start.component.css']
})
export class SeiteStartComponent {
  currentSubRoute: string = 'none';
  myObserver: any

  constructor(private router: Router,) { 
    this.myObserver = (this.router.events.pipe(filter((event: any) => event instanceof NavigationEnd)) as Observable<NavigationEnd>).subscribe((router: any) => {
      // console.log('------')
        this.currentSubRoute = router.urlAfterRedirects
    })
  }

  ngOnDestroy(): void {
    this.myObserver.unsubscribe()
  }
}
