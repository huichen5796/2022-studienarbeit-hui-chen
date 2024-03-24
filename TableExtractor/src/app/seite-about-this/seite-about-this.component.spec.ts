import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteAboutThisComponent } from './seite-about-this.component';

describe('SeiteAboutThisComponent', () => {
  let component: SeiteAboutThisComponent;
  let fixture: ComponentFixture<SeiteAboutThisComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteAboutThisComponent]
    });
    fixture = TestBed.createComponent(SeiteAboutThisComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
