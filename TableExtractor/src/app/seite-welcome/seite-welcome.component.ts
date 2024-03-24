import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-seite-welcome',
  templateUrl: './seite-welcome.component.html',
  styleUrls: ['./seite-welcome.component.css']
})
export class SeiteWelcomeComponent {
  @Input() side: boolean = false
  @Input() highlight: string = 'none'
  resourcesCards: { href: string, icon: string, span: string }[] = [
    { href: 'about-us', icon: 'perm_identity', span: 'About Us' },
    { href: 'about-this', icon: 'help_outline', span: 'About This' },
    { href: 'technology', icon: 'developer_board', span: 'Technology' },
    { href: 'database', icon: 'cloud_queue', span: 'Database' },
    { href: 'nn', icon: 'blur_on', span: 'Neural Networks' }
  ]

  functionCards: { icon: string, span: string, href: string }[] = [
    { icon: 'center_focus_weak', span: 'Table Exactor', href: 'start/main-exactor' },
    { icon: 'insert_chart_outlined', span: 'Coming soon', href: 'start/feedback' },
    { icon: 'storage', span: 'Coming soon', href: 'start/feedback' },
    { icon: 'bubble_chart', span: 'Guide', href: 'start/guide' },
    { icon: 'trending_up', span: 'Coming soon', href: 'start/feedback' },
    { icon: 'favorite_border', span: 'Feedback', href: 'start/feedback' }
  ]

  check(input:string){
    return `/${input}` === this.highlight
  }

}
