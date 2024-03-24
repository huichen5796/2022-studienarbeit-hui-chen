import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteWelcomeComponent } from './seite-welcome.component';

describe('SeiteWelcomeComponent', () => {
  let component: SeiteWelcomeComponent;
  let fixture: ComponentFixture<SeiteWelcomeComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteWelcomeComponent]
    });
    fixture = TestBed.createComponent(SeiteWelcomeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
